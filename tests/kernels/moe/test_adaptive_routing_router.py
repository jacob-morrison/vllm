# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AdaptiveRoutingRouter: real reduced dispatch for adaptive-compute K policies.

Needs the ``adaptive_routing`` package and a CUDA device (the fused top-k kernels).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.adaptive_routing_router import (
    AdaptiveRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.platforms import current_platform

adaptive_routing = pytest.importorskip("adaptive_routing")

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="fused top-k kernels need a GPU"
)

E, K0, M = 256, 8, 64


def _set_policy(monkeypatch, kind, k, **extra):
    monkeypatch.delenv("ADAPTIVE_ROUTING_MASKED", raising=False)
    monkeypatch.delenv("ADAPTIVE_ROUTING_NATIVE_K", raising=False)
    monkeypatch.setenv("ADAPTIVE_ROUTING_POLICY", kind)
    monkeypatch.setenv("ADAPTIVE_ROUTING_K", str(k))
    monkeypatch.setenv("ADAPTIVE_ROUTING_SPEC", "qwen3_5_moe")
    for key, value in extra.items():
        monkeypatch.setenv(key, str(value))


def _inputs():
    torch.manual_seed(0)
    logits = torch.randn(M, E, device="cuda") * 3
    hidden = torch.randn(M, 16, device="cuda")
    return hidden, logits


@pytest.mark.parametrize("kind", ["normalized", "reference"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_wrapper_matches_package_and_masked_rows(monkeypatch, kind, k):
    _set_policy(monkeypatch, kind, k)
    router = create_fused_moe_router(top_k=K0, global_num_experts=E)
    assert isinstance(router, AdaptiveRoutingRouter)
    assert router.top_k == k and not router.use_logits_path
    hidden, logits = _inputs()
    weights, ids = router.select_experts(hidden, logits)
    assert weights.shape == (M, k) and ids.shape == (M, k)

    policy = adaptive_routing.RoutingPolicy(kind, k)
    expected = adaptive_routing.route(logits, router.spec, policy)
    assert torch.equal(ids.long(), expected.ids.long())
    torch.testing.assert_close(weights.float(), expected.weights, rtol=1e-4, atol=1e-5)
    if kind == "normalized":
        torch.testing.assert_close(
            weights.float().sum(-1), torch.ones(M, device="cuda"), rtol=1e-4, atol=1e-5
        )

    monkeypatch.setenv("ADAPTIVE_ROUTING_MASKED", "1")
    masked = create_fused_moe_router(top_k=K0, global_num_experts=E)
    masked_w, masked_ids = masked.select_experts(hidden, logits)
    assert masked_w.shape == (M, K0)
    torch.testing.assert_close(masked_w[:, :k].float(), weights.float())
    assert torch.equal(masked_ids[:, :k].long(), ids.long())
    assert bool((masked_w[:, k:] == 0).all())


def test_native_policy_and_unset_return_the_plain_router(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_ROUTING_POLICY", raising=False)
    assert isinstance(
        create_fused_moe_router(top_k=K0, global_num_experts=E), FusedTopKRouter
    )
    monkeypatch.setenv("ADAPTIVE_ROUTING_POLICY", "native")
    assert isinstance(
        create_fused_moe_router(top_k=K0, global_num_experts=E), FusedTopKRouter
    )


def test_above_native_reference_widens_the_inner_router(monkeypatch):
    _set_policy(monkeypatch, "reference", 12)
    router = create_fused_moe_router(top_k=K0, global_num_experts=E)
    assert router.inner.top_k == 12 and router.native_k == K0
    hidden, logits = _inputs()
    weights, ids = router.select_experts(hidden, logits)
    assert weights.shape == (M, 12)
    native_w, native_ids = FusedTopKRouter(
        top_k=K0, global_num_experts=E, eplb_state=router.eplb_state
    ).select_experts(hidden, logits)
    assert torch.equal(ids[:, :K0].long(), native_ids.long())
    torch.testing.assert_close(
        weights[:, :K0].float(), native_w.float(), rtol=1e-4, atol=1e-5
    )
    assert bool((weights.float().sum(-1) > 1.0).all())


def test_hf_overrides_path_uses_env_native_k(monkeypatch):
    # --hf-overrides widened the router to 12; the native K comes from the env.
    _set_policy(monkeypatch, "reference", 12, ADAPTIVE_ROUTING_NATIVE_K=K0)
    router = create_fused_moe_router(top_k=12, global_num_experts=E)
    assert router.inner.top_k == 12 and router.native_k == K0 and router.top_k == 12
    hidden, logits = _inputs()
    weights, _ = router.select_experts(hidden, logits)
    torch.testing.assert_close(
        weights[:, :K0].float().sum(-1),
        torch.ones(M, device="cuda"),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("kind", ["normalized", "reference"])
def test_narrow_dispatch_matches_masked_through_fused_experts(monkeypatch, kind):
    """The kernel result with [M, k] rows equals the masked [M, K0] rows (old runs)."""
    experts, hidden_size, intermediate, tokens, k = 16, 128, 256, 32, 4
    native_k = 8
    torch.manual_seed(1)
    w1 = (
        torch.randn(
            experts, 2 * intermediate, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        / 10
    )
    w2 = (
        torch.randn(
            experts, hidden_size, intermediate, device="cuda", dtype=torch.bfloat16
        )
        / 10
    )
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(tokens, experts, device="cuda") * 2

    spec = adaptive_routing.spec_from_values(
        "qwen3_moe", native_k=native_k, num_experts=experts
    )
    narrow = adaptive_routing.route(
        logits, spec, adaptive_routing.RoutingPolicy(kind, k)
    )
    masked = adaptive_routing.route(
        logits, spec, adaptive_routing.RoutingPolicy(kind, k, masked=True)
    )
    assert narrow.ids.shape[-1] == k and masked.ids.shape[-1] == native_k

    out_narrow = fused_experts(x, w1, w2, narrow.weights, narrow.ids, inplace=False)
    out_masked = fused_experts(x, w1, w2, masked.weights, masked.ids, inplace=False)
    torch.testing.assert_close(out_narrow, out_masked, rtol=2e-2, atol=2e-2)


def test_fail_closed(monkeypatch):
    _set_policy(monkeypatch, "reference", 4)
    with pytest.raises(RuntimeError, match="custom routing"):
        create_fused_moe_router(
            top_k=K0, global_num_experts=E, custom_routing_function=lambda *a, **k: None
        )
    monkeypatch.setenv("ADAPTIVE_ROUTING_NUM_EXPERTS", "128")
    with pytest.raises(RuntimeError, match="experts"):
        create_fused_moe_router(top_k=K0, global_num_experts=E)
    monkeypatch.delenv("ADAPTIVE_ROUTING_NUM_EXPERTS")
    monkeypatch.setenv("ADAPTIVE_ROUTING_SPEC", "glm5")  # sigmoid + bias template
    with pytest.raises(RuntimeError, match="template"):
        create_fused_moe_router(top_k=K0, global_num_experts=E)
    monkeypatch.setenv("ADAPTIVE_ROUTING_SPEC", "qwen3_5_moe")
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random")
    with pytest.raises(RuntimeError, match="SIMULATION"):
        create_fused_moe_router(top_k=K0, global_num_experts=E)
