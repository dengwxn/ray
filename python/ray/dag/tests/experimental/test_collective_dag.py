# coding: utf-8
import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import pytest

import ray
import ray.experimental.collective as collective
from ray.dag import InputNode, InputAttributeNode, MultiOutputNode, ClassMethodNode
from ray.dag.compiled_dag_node import CompiledDAG
from ray.experimental.channel import CPUCommunicator
from ray.experimental.collective.conftest import (
    AbstractNcclGroup,
    CPUTorchTensorWorker,
    check_nccl_group_init,
    check_nccl_group_teardown,
)
from ray.experimental.util.types import ReduceOp
from ray.tests.conftest import *  # noqa

if TYPE_CHECKING:
    import cupy as cp
    import torch


logger = logging.getLogger(__name__)

if sys.platform != "linux" and sys.platform != "darwin":
    pytest.skip("Skipping, requires Linux or Mac.", allow_module_level=True)


class MockCommunicator(CPUCommunicator):
    """
    Use a mock communicator to test the actor schedules.
    """

    def __init__(self, world_size: int, actor_handles: List["ray.actor.ActorHandle"]):
        self._world_size = world_size
        self._actor_handles = actor_handles

    def send(self, value: "torch.Tensor", peer_rank: int) -> None:
        raise NotImplementedError

    def recv(
        self,
        shape: Tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator: Optional[
            Callable[[Tuple[int], "torch.dtype"], "torch.Tensor"]
        ] = None,
    ) -> "torch.Tensor":
        raise NotImplementedError

    def allgather(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
    ) -> None:
        raise NotImplementedError

    def allreduce(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp,
    ) -> None:
        raise NotImplementedError

    def reducescatter(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp,
    ) -> None:
        raise NotImplementedError

    @property
    def recv_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        raise NotImplementedError

    @property
    def send_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        raise NotImplementedError

    @property
    def coll_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        raise NotImplementedError

    def destroy(self) -> None:
        raise NotImplementedError


@ray.remote
class DDPWorker:
    def __init__(self):
        return

    def backward(self, _):
        return 0


@ray.remote
class P2PWorker:
    def __init__(self):
        pass

    def send(self, _) -> "torch.Tensor":
        import torch

        return torch.ones(10)

    def recv(self, tensor: "torch.Tensor") -> int:
        return tensor[0]


@pytest.mark.parametrize("ray_start_regular", [{"num_cpus": 4}], indirect=True)
def test_all_reduce_duplicate_actors(ray_start_regular):
    """
    Test an error is thrown when two input nodes from the same actor bind to
    an all-reduce.
    """
    actor_cls = CPUTorchTensorWorker.options()
    worker = actor_cls.remote()

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for _ in range(2)]
        with pytest.raises(
            ValueError,
            match="Expected unique actor handles for a collective operation",
        ):
            collective.allreduce.bind(computes)

    with InputNode() as inp:
        compute = worker.return_tensor.bind(inp)
        computes = [compute for _ in range(2)]
        with pytest.raises(
            ValueError,
            match="Expected unique input nodes for a collective operation",
        ):
            collective.allreduce.bind(computes)


@pytest.mark.parametrize("ray_start_regular", [{"num_cpus": 4}], indirect=True)
def test_all_reduce_custom_comm_wrong_actors(ray_start_regular):
    """
    Test an error is thrown when an all-reduce binds to a custom NCCL group and
    a wrong set of actors.
    """
    actor_cls = CPUTorchTensorWorker.options()

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    nccl_group = AbstractNcclGroup([workers[0]])
    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        with pytest.raises(
            ValueError,
            match="Expected actor handles to match the custom NCCL group",
        ):
            collective.allreduce.bind(computes, transport=nccl_group)


@pytest.mark.parametrize(
    "ray_start_regular", [{"num_cpus": 4, "num_gpus": 4}], indirect=True
)
def test_comm_all_reduces(ray_start_regular, monkeypatch):
    """
    Test different communicators are used for different all-reduce calls of
    different sets of actors.
    """
    actor_cls = CPUTorchTensorWorker.options(num_cpus=0, num_gpus=1)

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        # There are two all-reduces, each on one actor.
        collectives = [collective.allreduce.bind([compute]) for compute in computes]
        # collective[0] is the only CollectiveOutputNode for each all-reduce.
        dag = MultiOutputNode([collective[0] for collective in collectives])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {
            (frozenset([workers[0]]), None),
            (frozenset([workers[1]]), None),
        },
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)


@pytest.mark.parametrize(
    "ray_start_regular", [{"num_cpus": 4, "num_gpus": 4}], indirect=True
)
def test_comm_deduplicate_all_reduces(ray_start_regular, monkeypatch):
    """
    Test communicators are deduplicated when all-reduces are called on the same
    group of actors more than once.
    """
    actor_cls = CPUTorchTensorWorker.options(num_cpus=0, num_gpus=1)

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    with InputNode() as inp:
        tensors = [worker.return_tensor.bind(inp) for worker in workers]
        collectives = collective.allreduce.bind(tensors)
        collectives = collective.allreduce.bind(collectives)
        dag = MultiOutputNode(collectives)

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {(frozenset(workers), None)},
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)


@pytest.mark.parametrize(
    "ray_start_regular", [{"num_cpus": 4, "num_gpus": 4}], indirect=True
)
def test_comm_deduplicate_p2p_and_collective(ray_start_regular, monkeypatch):
    """
    Test communicators are deduplicated when the collective and the P2P are on
    the same set of actors.
    """
    actor_cls = CPUTorchTensorWorker.options(num_cpus=0, num_gpus=1)

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        collectives = collective.allreduce.bind(computes)
        recvs = [
            # Each of the 2 workers receives from the other.
            workers[0].recv.bind(
                collectives[1].with_tensor_transport(transport="nccl")
            ),
            workers[1].recv.bind(
                collectives[0].with_tensor_transport(transport="nccl")
            ),
        ]
        dag = MultiOutputNode(recvs)

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {(frozenset(workers), None)},
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        collectives = collective.allreduce.bind(computes)
        # Sender is workers[0] and receiver is workers[1].
        dag = workers[1].recv.bind(
            collectives[0].with_tensor_transport(transport="nccl")
        )
        dag = MultiOutputNode([dag, collectives[1]])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {(frozenset(workers), None)},
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)


@pytest.mark.parametrize(
    "ray_start_regular", [{"num_cpus": 4, "num_gpus": 4}], indirect=True
)
def test_custom_comm(ray_start_regular, monkeypatch):
    """
    Test a custom GPU communicator is used when specified and a default
    communicator is used otherwise.
    """
    actor_cls = CPUTorchTensorWorker.options(num_cpus=0, num_gpus=1)

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    comm = AbstractNcclGroup(workers)
    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        collectives = collective.allreduce.bind(computes, transport=comm)
        collectives = collective.allreduce.bind(collectives)
        dag = workers[0].recv.bind(
            collectives[1].with_tensor_transport(transport="nccl")
        )
        dag = MultiOutputNode([dag, collectives[0]])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {
            (frozenset(workers), comm),
            (frozenset(workers), None),
        },
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)

    comm = AbstractNcclGroup(workers)
    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers]
        collectives = collective.allreduce.bind(computes)
        collectives = collective.allreduce.bind(collectives)
        dag = workers[0].recv.bind(collectives[1].with_tensor_transport(transport=comm))
        dag = MultiOutputNode([dag, collectives[0]])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {
            (frozenset(workers), comm),
            (frozenset(workers), None),
        },
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)


@pytest.mark.parametrize(
    "ray_start_regular", [{"num_cpus": 4, "num_gpus": 4}], indirect=True
)
def test_custom_comm_init_teardown(ray_start_regular, monkeypatch):
    """
    Test custom NCCL groups are properly initialized and destroyed.
    1. Test when multiple type hints have the same `transport=custom_nccl_group`,
    the `custom_nccl_group` is initialized only once.
    2. Test all initialized NCCL groups are destroyed during teardown.
    """
    actor_cls = CPUTorchTensorWorker.options(num_cpus=0, num_gpus=1)

    num_workers = 2
    workers = [actor_cls.remote() for _ in range(num_workers)]

    comm = AbstractNcclGroup(workers)

    with InputNode() as inp:
        tensors = [worker.return_tensor.bind(inp) for worker in workers]
        allreduce = collective.allreduce.bind(tensors, transport=comm)
        dag = workers[0].recv.bind(allreduce[1].with_tensor_transport(transport=comm))
        dag = MultiOutputNode([dag, allreduce[0]])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {(frozenset(workers), comm)},
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)

    comm_1 = AbstractNcclGroup(workers)
    comm_2 = AbstractNcclGroup(workers)
    comm_3 = AbstractNcclGroup(workers)

    with InputNode() as inp:
        tensors = [worker.return_tensor.bind(inp) for worker in workers]
        allreduce1 = collective.allreduce.bind(tensors, transport=comm_1)
        allreduce2 = collective.allreduce.bind(allreduce1, transport=comm_2)
        dag = workers[0].recv.bind(
            allreduce2[1].with_tensor_transport(transport=comm_3)
        )
        dag = MultiOutputNode([dag, allreduce2[0]])

    compiled_dag, mock_nccl_group_set = check_nccl_group_init(
        monkeypatch,
        dag,
        {
            (frozenset(workers), comm_1),
            (frozenset(workers), comm_2),
            (frozenset(workers), comm_3),
        },
    )

    check_nccl_group_teardown(monkeypatch, compiled_dag, mock_nccl_group_set)


@pytest.mark.parametrize("ray_start_regular", [{"num_cpus": 4}], indirect=True)
@pytest.mark.parametrize("num_workers", [2, 4])
def test_exec_schedules_ddp(ray_start_regular, num_workers):
    """
    Test the execution schedules for the DDP strategy. Each worker should have
    identical schedules.
    """
    actor_cls = DDPWorker.options(num_cpus=1)
    workers = [actor_cls.remote() for _ in range(num_workers)]
    comm = MockCommunicator(num_workers, workers)

    outputs = []
    with InputNode() as inp:
        grads = [worker.backward.bind(inp) for worker in workers]
        grads_reduced = collective.allreduce.bind(grads, transport=comm)
        outputs.extend(grads_reduced)
        grads = [worker.backward.bind(grad) for worker, grad in zip(workers, grads)]
        grads_reduced = collective.allreduce.bind(grads, transport=comm)
        outputs.extend(grads_reduced)
        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile(_default_communicator=comm)
    actor_to_execution_schedule = list(
        compiled_dag.actor_to_execution_schedule.values()
    )
    expected_schedule = actor_to_execution_schedule[0]
    for schedule in actor_to_execution_schedule[1:]:
        assert schedule == expected_schedule


@pytest.mark.parametrize("ray_start_regular", [{"num_cpus": 4}], indirect=True)
def test_add_p2p_nodes(ray_start_regular):
    p2p_worker = P2PWorker.options(num_cpus=1).remote()
    input_node = InputNode()
    with input_node:
        send_node = p2p_worker.send.bind(input_node)
        send_node.with_tensor_transport(transport="nccl")
        recv_node = p2p_worker.recv.bind(send_node)

    compiled_dag = CompiledDAG()
    node_to_p2p_send_node: Dict["ray.dag.DAGNode", "ray.dag.P2PSendNode"] = {}
    compiled_dag._add_p2p_recv_nodes(input_node, node_to_p2p_send_node)
    compiled_dag._add_node(input_node)
    compiled_dag._add_p2p_send_node(input_node, node_to_p2p_send_node)
    # The input node is added, so there is 1 task in the compiled DAG.
    assert len(compiled_dag.idx_to_task) == 1
    assert node_to_p2p_send_node == {}

    compiled_dag._add_p2p_recv_nodes(send_node, node_to_p2p_send_node)
    compiled_dag._add_node(send_node)
    assert len(compiled_dag.idx_to_task) == 2
    assert node_to_p2p_send_node == {}
    compiled_dag._add_p2p_send_node(send_node, node_to_p2p_send_node)
    # A P2PSendNode is added.
    assert len(compiled_dag.idx_to_task) == 3
    # The dictionary becomes {send_node: P2PSendNode}.
    assert len(node_to_p2p_send_node) == 1
    assert send_node in node_to_p2p_send_node

    compiled_dag._add_p2p_recv_nodes(recv_node, node_to_p2p_send_node)
    # A P2PRecvNode is added.
    assert len(compiled_dag.idx_to_task) == 4
    compiled_dag._add_node(recv_node)
    compiled_dag._add_p2p_send_node(recv_node, node_to_p2p_send_node)
    assert len(compiled_dag.idx_to_task) == 5
    # Adding the recv node and its P2PRecvNode does not change the dictionary.
    assert len(node_to_p2p_send_node) == 1

    compiled_dag.teardown()


@pytest.mark.parametrize("ray_start_regular", [{"num_cpus": 4}], indirect=True)
def test_add_p2p_nodes_exceptions(ray_start_regular):
    p2p_worker = P2PWorker.options(num_cpus=1).remote()
    compiled_dag = CompiledDAG()

    # InputNode cannot be a P2P sender.
    input_node = InputNode()
    input_node.with_tensor_transport(transport="nccl")
    with pytest.raises(
        ValueError,
        match="DAG inputs cannot be transferred via NCCL because the driver "
        "cannot participate in the NCCL group",
    ):
        compiled_dag._add_p2p_send_node(input_node, {})

    # Dag output node cannot be a P2P sender.
    output_node = p2p_worker.recv.bind(None)
    output_node.with_tensor_transport(transport="nccl")
    output_node.is_cgraph_output_node = True
    with pytest.raises(
        ValueError,
        match="Outputs cannot be transferred via NCCL because the driver "
        "cannot participate in the NCCL group",
    ):
        compiled_dag._add_p2p_send_node(output_node, {})

    @ray.remote
    def fn(*args):
        return 0

    # Non-ClassMethodNode cannot be a P2P sender.
    function_node = fn.bind().with_tensor_transport(transport="nccl")
    with pytest.raises(
        ValueError,
        match="NCCL P2P operation is only supported with ClassMethodNode",
    ):
        compiled_dag._add_p2p_send_node(function_node, {})

    # MultiOutputNode cannot be a P2P receiver.
    send_node = p2p_worker.send.bind(None)
    send_node.with_tensor_transport(transport="nccl")
    multi_output_node = MultiOutputNode([send_node])
    # Adding the send node does not error.
    node_to_p2p_send_node: Dict["ray.dag.DAGNode", "ray.dag.P2PSendNode"] = {}
    compiled_dag._add_p2p_send_node(send_node, node_to_p2p_send_node)
    # Adding the MultiOutputNode as a P2P receiver errors.
    with pytest.raises(
        ValueError,
        match="Outputs cannot be transferred via NCCL because the driver "
        "cannot participate in the NCCL group",
    ):
        compiled_dag._add_p2p_recv_nodes(multi_output_node, node_to_p2p_send_node)
    compiled_dag.teardown()

    # Non-ClassMethodNode cannot be a P2P receiver.
    send_node = p2p_worker.send.bind(None)
    send_node.with_tensor_transport(transport="nccl")
    function_node = fn.bind(send_node)
    compiled_dag = CompiledDAG()
    node_to_p2p_send_node: Dict["ray.dag.DAGNode", "ray.dag.P2PSendNode"] = {}
    compiled_dag._add_p2p_send_node(send_node, node_to_p2p_send_node)
    with pytest.raises(
        ValueError,
        match="NCCL P2P operation is only supported with ClassMethodNode",
    ):
        compiled_dag._add_p2p_recv_nodes(function_node, node_to_p2p_send_node)
    compiled_dag.teardown()


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
