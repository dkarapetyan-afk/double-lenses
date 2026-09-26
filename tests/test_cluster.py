"""
Distributed Cluster Staging Tests:
  - Topology & Heterogeneous Device Addressing
  - Serializer & Communication Fabric
  - Collectives & Adjoint Duals (AllGather* = ReduceScatter, etc.)
  - Tensor Parallelism & Expert Parallelism 2-Cells
  - DistributedLensRuntime
"""

import numpy as np

from double_lenses.autodiff.loss import CrossEntropyLossLens
from double_lenses.autodiff.param_lens import LensContext
from double_lenses.autodiff.tensor import randn
from double_lenses.cluster.collectives import (
    AllGatherFunctor,
    AllReduceFunctor,
)
from double_lenses.cluster.engine import DistributedLensRuntime
from double_lenses.cluster.fabric import CommunicationFabric, Serializer
from double_lenses.cluster.sharding import ExpertParallel2Cell, TensorParallel2Cell
from double_lenses.cluster.topology import ClusterTopology
from double_lenses.models.config import DeepSeekConfig, MixtralConfig
from double_lenses.models.deepseek import DeepSeekModelLens
from double_lenses.models.mixtral import MixtralModelLens


def test_tensor_serialization():
    t = randn((3, 5))
    tag = "test_tag"
    buf = Serializer.serialize_tensor(t, tag=tag)
    deser_t, deser_tag = Serializer.deserialize_tensor(buf)

    assert deser_tag == tag
    assert deser_t.shape == t.shape
    np.testing.assert_allclose(deser_t.to_numpy(), t.to_numpy())


def test_collectives_and_adjoint_duals():
    top = ClusterTopology.create_local_heterogeneous(num_gpus=1, num_cpus=2)
    fabric = CommunicationFabric(top)
    devs = top.all_devices()[:3]  # 3 devices

    # 1. AllReduce
    ar = AllReduceFunctor(fabric, devs)
    inputs = {str(d): randn((2, 4)) for d in devs}
    out = ar.forward(inputs)

    expected_sum = sum(inputs[str(d)].to_numpy() for d in devs)
    for d in devs:
        np.testing.assert_allclose(out[str(d)].to_numpy(), expected_sum)

    # AllReduce adjoint is AllReduce
    adj_out = ar.adjoint(out)
    for d in devs:
        np.testing.assert_allclose(adj_out[str(d)].to_numpy(), expected_sum * 3.0)

    # 2. AllGather & ReduceScatter adjoint
    ag = AllGatherFunctor(fabric, devs, axis=0)
    shards = {str(d): randn((2, 4)) for d in devs}
    gathered = ag.forward(shards)
    expected_cat = np.concatenate([shards[str(d)].to_numpy() for d in devs], axis=0)

    for d in devs:
        np.testing.assert_allclose(gathered[str(d)].to_numpy(), expected_cat)

    # Adjoint dual (ReduceScatter)
    grad_shards = ag.adjoint(gathered)
    for d in devs:
        assert grad_shards[str(d)].shape == (2, 4)


def test_tensor_parallel_2cell():
    top = ClusterTopology.create_local_heterogeneous(num_gpus=1, num_cpus=2)
    fabric = CommunicationFabric(top)
    devs = [top.all_devices()[0], top.all_devices()[1]]  # 2 shards

    tp_cell = TensorParallel2Cell(
        name="tp_cell",
        in_features=8,
        hidden_features=16,
        out_features=8,
        devices=devs,
        fabric=fabric,
    )

    x = randn((2, 8))
    ctx = LensContext()
    out = tp_cell.forward(ctx, x)
    assert out.shape == (2, 8)

    dy = randn((2, 8))
    dx = tp_cell.adjoint(ctx, dy)
    assert dx.shape == (2, 8)


def test_expert_parallel_2cell():
    top = ClusterTopology.create_local_heterogeneous(num_gpus=1, num_cpus=2)
    fabric = CommunicationFabric(top)
    all_devs = top.all_devices()

    # Map 4 experts across available devices
    placements = {
        0: all_devs[0],
        1: all_devs[1],
        2: all_devs[2],
        3: all_devs[0],
    }

    ep_cell = ExpertParallel2Cell(
        name="ep_cell",
        dim=8,
        hidden_dim=16,
        num_experts=4,
        top_k=2,
        expert_placements=placements,
        fabric=fabric,
        ingress_device=all_devs[0],
    )

    n_tokens = 4
    flat_x = np.random.randn(n_tokens, 8).astype(np.float32)
    flat_weights = np.ones((n_tokens, 2), dtype=np.float32) * 0.5
    flat_experts = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)

    ctx = LensContext()
    out = ep_cell.forward(ctx, flat_x, flat_weights, flat_experts)
    assert out.shape == (n_tokens, 8)

    # Distributed adjoint pass
    dy = np.random.randn(n_tokens, 8).astype(np.float32)
    grad_x, grad_w = ep_cell.adjoint(ctx, dy, flat_weights)
    assert grad_x.shape == flat_x.shape
    assert grad_w.shape == flat_weights.shape


def test_distributed_runtime_end_to_end():
    top = ClusterTopology.create_local_heterogeneous(num_gpus=1, num_cpus=2)
    runtime = DistributedLensRuntime(top)
    runtime.start()

    try:
        # Test Mixtral staging and execution
        m_config = MixtralConfig.mixtral_mini()
        m_model = MixtralModelLens("mixtral", m_config)
        runtime.stage_mixtral(m_model)

        loss_lens = CrossEntropyLossLens("loss")
        tokens = np.random.randint(0, m_config.vocab_size, size=(2, 4))
        targets = np.random.randint(0, m_config.vocab_size, size=(2, 4))

        loss, stats = runtime.run_step(m_model, loss_lens, tokens, targets)
        assert loss > 0.0
        assert stats["forward_passes"] >= 1
        assert stats["adjoint_passes"] >= 1

        # Test DeepSeek staging and execution
        d_config = DeepSeekConfig.deepseek_mini()
        d_model = DeepSeekModelLens("deepseek", d_config)
        runtime.stage_deepseek(d_model)

        loss_d, stats_d = runtime.run_step(d_model, loss_lens, tokens, targets)
        assert loss_d > 0.0
    finally:
        runtime.stop()
