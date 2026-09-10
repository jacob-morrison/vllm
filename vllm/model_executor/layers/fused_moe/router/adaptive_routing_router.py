# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive-compute K policies with real reduced dispatch.

Opt-in through the environment (``ADAPTIVE_ROUTING_POLICY`` = ``normalized`` or
``reference``, ``ADAPTIVE_ROUTING_K``; see the ``adaptive_routing`` package for the
full list; ``ADAPTIVE_ROUTING_FROM_LOGITS=1`` forces re-selection from the logits
instead of the post-hoc transform of the inner router's rows). When active,
``create_fused_moe_router`` wraps the router it would have built in an
:class:`AdaptiveRoutingRouter`, which returns ``[M, K]`` tensors for the
requested K. The fused-experts kernels read K from the tensor width, so fewer
experts are actually dispatched; the weights follow the policy (renormalized over
the K survivors, or the native-K denominator kept).

Fails closed: monolithic kernels (FlashInfer TRT-LLM, CPU) route inside the kernel
and never call the router; custom routing functions have no spec; above-native K
with expert parallelism would overflow all-to-all buffers sized from the config K.
"""

from __future__ import annotations

import atexit
import os
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)

try:
    import adaptive_routing as _ar
except ImportError:  # the package is optional; only needed when a policy is set
    _ar = None

POLICY_ENV = "ADAPTIVE_ROUTING_POLICY"


def requested_policy() -> Any | None:
    """The policy requested through ``ADAPTIVE_ROUTING_*``, or None when unset."""
    if not os.environ.get(POLICY_ENV, "").strip():
        return None
    if _ar is None:
        raise ImportError(
            f"{POLICY_ENV} is set but the adaptive_routing package is not "
            "installed in the vLLM environment"
        )
    return _ar.policy_from_env()


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_tracker: Any | None = None


def _get_tracker() -> Any:
    global _tracker
    if _tracker is None:
        assert _ar is not None
        _tracker = _ar.RealizedKTracker(max_k=64)
        atexit.register(_log_summary)
    return _tracker


def _log_summary() -> None:
    if _tracker is not None and _tracker.tokens:
        logger.info("%s", _tracker.summary())


def build_spec(
    *,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    has_bias: bool,
    use_grouped_topk: bool,
    num_expert_group: int | None,
    topk_group: int | None,
) -> Any:
    """Spec from the router arguments, cross-checked against the environment.

    ``ADAPTIVE_ROUTING_SPEC`` names a family template whose score function,
    renormalization and bias must match the router; ``ADAPTIVE_ROUTING_NATIVE_K``
    and ``ADAPTIVE_ROUTING_NUM_EXPERTS`` pin the checkpoint constants. The layer's
    ``top_k`` is the native K unless ``ADAPTIVE_ROUTING_NATIVE_K`` says otherwise
    (the ``--hf-overrides`` path for above-native K widens the router itself).
    """
    assert _ar is not None
    overrides = _ar.spec_overrides_from_env()
    native_k = int(overrides.get("native_k", top_k))
    num_experts = int(overrides.get("num_experts", global_num_experts))
    if num_experts != global_num_experts:
        raise RuntimeError(
            f"adaptive routing: router has {global_num_experts} experts but "
            f"ADAPTIVE_ROUTING_NUM_EXPERTS={num_experts}"
        )
    if native_k > top_k:
        raise RuntimeError(
            f"adaptive routing: ADAPTIVE_ROUTING_NATIVE_K={native_k} exceeds the "
            f"router top_k={top_k}"
        )
    values = dict(
        score_fn=scoring_func,
        native_k=native_k,
        num_experts=num_experts,
        renormalize=bool(renormalize),
        scale=float(routed_scaling_factor),
        selection_bias=bool(has_bias),
        num_expert_group=int(num_expert_group or 1) if use_grouped_topk else 1,
        topk_group=int(topk_group or 1) if use_grouped_topk else 1,
        source="vllm-router-args",
    )
    family = overrides.get("family")
    if family:
        base = _ar.template(family)
        mismatches = {
            name: (values[name], getattr(base, name))
            for name in ("score_fn", "renormalize", "selection_bias")
            if values[name] != getattr(base, name)
        }
        if mismatches:
            raise RuntimeError(
                f"adaptive routing: router disagrees with the {base.family} "
                f"template on {mismatches}"
            )
        return replace(base, verified=True, **values)
    return _ar.RoutingSpec(family="runtime", **values)


class AdaptiveRoutingRouter(BaseRouter):
    """Wraps a router and applies an ``adaptive_routing`` policy.

    Two paths. For a plain fused top-k router of a renormalizing family the inner
    router runs at width ``max(K, native_k)`` and the sorted rows are transformed
    post hoc (``reweight_selected``), which reproduces the old masked patches
    bit-for-bit on the surviving experts. For biased or grouped routers the
    selection is recomputed from the logits (``route``).
    """

    _layer_counter = 0

    def __init__(
        self,
        inner: BaseRouter,
        spec: Any,
        policy: Any,
        *,
        eplb_state: EplbLayerState,
        enable_eplb: bool,
        indices_type_getter: Callable[[], torch.dtype | None] | None,
        e_score_correction_bias: torch.Tensor | None,
    ):
        self.k = policy.resolve_k(spec)
        super().__init__(
            top_k=self.k,
            global_num_experts=spec.num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.inner = inner
        self.spec = spec
        self.policy = policy
        self.e_score_correction_bias = e_score_correction_bias
        self.native_k = spec.native_k
        self.use_logits_path = (
            not (
                isinstance(inner, FusedTopKRouter)
                and spec.renormalize
                and not spec.selection_bias
            )
            or os.environ.get("ADAPTIVE_ROUTING_FROM_LOGITS", "0") == "1"
        )
        self.expected_width = max(self.k, self.native_k) if policy.masked else self.k
        self.layer_index = AdaptiveRoutingRouter._layer_counter
        AdaptiveRoutingRouter._layer_counter += 1
        self.calls = 0
        self.telemetry = os.environ.get("ADAPTIVE_ROUTING_TELEMETRY", "1") != "0"
        if self.layer_index == 0:
            logger.info(
                "adaptive routing: %s with %s; %s path; dispatch width %d",
                policy.describe(spec),
                spec.describe(),
                "logits" if self.use_logits_path else "post-hoc",
                self.expected_width,
            )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self.inner.routing_method_type

    def check_layer(self, layer: FusedMoE) -> None:
        """Startup guards that need the layer (called from ``FusedMoE.__init__``)."""
        if layer.quant_method.is_monolithic:
            raise RuntimeError(
                "adaptive routing: the MoE kernel is monolithic (routes inside the "
                "kernel, e.g. FlashInfer TRT-LLM or CPU) so the router is never "
                "called; pick a non-monolithic backend or unset "
                f"{POLICY_ENV}"
            )
        if layer.custom_routing_function is not None:
            raise RuntimeError(
                "adaptive routing: this model uses a custom routing function; "
                "no routing spec applies"
            )
        if self.k > self.native_k and layer.use_ep:
            raise RuntimeError(
                "adaptive routing: above-native K with expert parallelism sizes "
                "all-to-all buffers from the config K; use TP/DP"
            )
        if self.layer_index == 0:
            logger.info(
                "adaptive routing: kernel %s, use_ep=%s, layer top_k=%d, config "
                "experts_per_token=%d",
                type(layer.quant_method).__name__,
                layer.use_ep,
                layer.top_k,
                layer.moe_config.experts_per_token,
            )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert _ar is not None
        first = self.calls == 0 and not _is_capturing()
        if self.use_logits_path:
            bias = (
                None
                if self.e_score_correction_bias is None
                else self.e_score_correction_bias.data
            )
            out = _ar.route(router_logits, self.spec, self.policy, selection_bias=bias)
            topk_weights, topk_ids = out.weights, out.ids
        else:
            topk_weights, topk_ids = self.inner._compute_routing(
                hidden_states, router_logits, indices_type
            )
            if first:
                _ar.assert_sorted_descending(
                    topk_weights, where=f"layer {self.layer_index}"
                )
            topk_weights, topk_ids = _ar.reweight_selected(
                topk_weights, topk_ids, self.spec, self.policy
            )
        if first:
            _ar.assert_width(
                topk_ids, self.expected_width, where=f"layer {self.layer_index}"
            )
            if self.layer_index == 0:
                logger.info(
                    "adaptive routing: first batch ok, %d tokens, width %d",
                    topk_ids.shape[0],
                    topk_ids.shape[-1],
                )
        if self.telemetry:
            _get_tracker().record_fixed(self.layer_index, self.k, topk_ids.shape[0])
        self.calls += 1
        return topk_weights, topk_ids


def build_adaptive_router(
    *,
    policy: Any,
    inner_factory: Callable[..., BaseRouter],
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    indices_type_getter: Callable[[], torch.dtype | None] | None,
    use_grouped_topk: bool,
    num_expert_group: int | None,
    topk_group: int | None,
    scoring_func: str,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    custom_routing_function: Callable | None,
    enable_eplb: bool,
    eplb_state: EplbLayerState,
) -> BaseRouter:
    """Build the inner router at the width the policy needs and wrap it."""
    if envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY != "":
        raise RuntimeError(
            "adaptive routing cannot be combined with "
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
        )
    if custom_routing_function is not None:
        raise RuntimeError(
            "adaptive routing: this model uses a custom routing function; "
            "no routing spec applies"
        )
    spec = build_spec(
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        has_bias=e_score_correction_bias is not None,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )
    k = policy.resolve_k(spec)
    width = max(k, spec.native_k, top_k)
    inner = inner_factory(
        top_k=width,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        indices_type_getter=indices_type_getter,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        custom_routing_function=None,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )
    if not isinstance(inner, BaseRouter):
        raise RuntimeError(
            f"adaptive routing: cannot wrap router type {type(inner).__name__}"
        )
    return AdaptiveRoutingRouter(
        inner,
        spec,
        policy,
        eplb_state=eplb_state,
        enable_eplb=enable_eplb,
        indices_type_getter=indices_type_getter,
        e_score_correction_bias=e_score_correction_bias,
    )
