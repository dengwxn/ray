from functools import total_ordering
from ray.dag.nccl_operation import _NcclOperation
from ray.experimental.util.types import _NcclOp, P2POp, _CollectiveOp
from typing import Set, Tuple, List, Dict, Optional
import copy
import logging
import ray
import heapq
from collections import defaultdict


logger = logging.getLogger(__name__)


class _DAGNodeOperation:
    def __init__(
        self,
        exec_task_idx: int,
        method_name: Optional[str] = None,
    ):
        """
        Args:
            exec_task_idx: The index of the task that this operation belongs to
                in the actor's ExecutableTask list. The index is not the same
                as bind_index because there may be more tasks bound to an actor
                than tasks that appear in the current compiled DAG.
            method_name: The name of the method that this operation originates
                from. This is only for visualization and debugging purposes.
        """
        self.exec_task_idx = exec_task_idx
        self.method_name = method_name

    def __repr__(self):
        return (
            f"_DAGNodeOperation(exec_task_idx: {self.exec_task_idx}, "
            f"method_name: {self.method_name})"
        )

    def viz_str(self):
        """
        A string representation of the node to be used in visualization.
        """
        return f"([{self.exec_task_idx}] {self.method_name})"

    def __hash__(self):
        return hash(self.exec_task_idx)

    def __eq__(self, other: "_DAGNodeOperation"):
        # An operation is uniquely identified by its `exec_task_idx`.
        # `method_name` is only for debugging purposes.
        return self.exec_task_idx == other.exec_task_idx


@total_ordering
class _DAGOperationGraphNode:
    def __init__(
        self,
        op: _DAGNodeOperation,
        task_idx: int,
        actor_handle: "ray.actor.ActorHandle",
        nccl_op: Optional[_NcclOperation] = None,
        nccl_op_type: Optional[_NcclOp] = None,
    ):
        """
        _DAGOperationGraphNode represents a node in the DAG operation graph.
        It contains information about the node's in-degree, out-degree, edges,
        and the operation it performs.

        Args:
            op: The operation that this node performs.
            task_idx: A unique index which can be used to index into
                `CompiledDAG.idx_to_task` to get the corresponding task.
            actor_handle: The actor handle to which this operation belongs.
            requires_nccl: Whether this operation requires NCCL.
        """
        self.op = op
        self.task_idx = task_idx
        self.actor_handle = actor_handle
        self.nccl_op_type = nccl_op_type
        # The in_edges and out_edges are dicts of ints to strings. Each int (the key)
        # is an integer `task_idx`, which can be used to index into `idx_to_task` to
        # get the corresponding task. The string (the value) is the visualization
        # information of the edge, it is a tuple of a label of the edge and a boolean
        # indicating whether the edge is a control dependency.
        self.in_edges: Dict[int, Tuple[str, bool]] = {}
        self.out_edges: Dict[int, Tuple[str, bool]] = {}
        # The NCCL operation of the task. It can be a NCCL read, write, or
        # collective operation.
        self.nccl_op: Optional[_NcclOperation] = nccl_op
        # The input task index, if this is a NCCL collective operation.
        self.nccl_collective_input_idx: Optional[int] = None

    def __repr__(self):
        return (
            f"_DAGOperationGraphNode("
            f"op: {self.op}, "
            f"task_idx: {self.task_idx}, "
            f"ray_actor_id: {self.actor_handle._ray_actor_id}, "
            f"nccl_op_type: {self.nccl_op_type})"
        )

    def __lt__(self, other: "_DAGOperationGraphNode"):
        """
        This function defines the order of the nodes in the priority queue used in
        `_select_next_nodes`. The priority queue is a min-heap, so the node with
        higher priority is considered "less than" the other node.
        """

        def compare(lhs: "_DAGOperationGraphNode", rhs: "_DAGOperationGraphNode"):
            # If both nodes belong to the same actor, the node with the smaller
            # `exec_task_idx` is prioritized. If two nodes belong to different
            # actors, it approximates balancing the scheduled tasks across actors,
            # by prioritizing the node with the smaller `exec_task_idx`. The tie
            # is broken by the `task_idx`.
            if lhs.op.exec_task_idx != rhs.op.exec_task_idx:
                return lhs.op.exec_task_idx < rhs.op.exec_task_idx
            return lhs.task_idx < rhs.task_idx

        if self.actor_handle == other.actor_handle:
            # When both nodes belong to the same actor, use the default comparison.
            return compare(self, other)
        elif self.requires_nccl_op != other.requires_nccl_op:
            # When one node is a NCCL operation and the other is not, prioritize
            # the non-NCCL operation.
            return not self.requires_nccl_op
        else:
            # When either both nodes are NCCL operations or both nodes are not
            # NCCL operations, use the default comparison.
            return compare(self, other)

    def __eq__(self, other: "_DAGOperationGraphNode"):
        """
        Two operations are equal only when they have the same `exec_task_idx`
        and belong to the same actor.
        """
        return (
            self.actor_handle == other.actor_handle
            and self.op.exec_task_idx == other.op.exec_task_idx
        )

    def __hash__(self):
        """
        An operation is uniquely identified by its `task_idx`.
        """
        return hash(self.task_idx)

    @property
    def in_degree(self) -> int:
        return len(self.in_edges)

    @property
    def is_ready(self) -> bool:
        """
        If a node is not a NCCL operation, it is ready when it has a zero in-degree.
        If it is a NCCL operation, it is ready when all the nodes in the operation
        have zero in-degrees.
        """
        return self.in_degree == 0 and (self.nccl_op is None or self.nccl_op.is_ready)

    @property
    def requires_nccl_op(self) -> bool:
        return self.nccl_op_type is not None

    @property
    def requires_nccl_read(self) -> bool:
        return self.nccl_op_type == P2POp.RECV

    @property
    def requires_nccl_write(self) -> bool:
        return self.nccl_op_type == P2POp.SEND

    @property
    def requires_nccl_collective(self) -> bool:
        return isinstance(self.nccl_op_type, _CollectiveOp)

    def viz_str(self):
        """
        A string representation of the node to be used in visualization.
        """
        return self.op.viz_str()

    @property
    def _actor_id(self):
        return self.actor_handle._ray_actor_id.hex()


def _add_edge(
    from_node: _DAGOperationGraphNode,
    to_node: _DAGOperationGraphNode,
    label: str = "",
    control_dependency: bool = False,
):
    """
    Add an edge from `from_node` to `to_node`.

    Args:
        from_node: The node from which the edge originates.
        to_node: The node to which the edge points.
        label: The label of the edge. This will be used to annotate the edge
            in the visualization of the execution schedule.
        control_dependency: Whether the edge represents a control dependency.
    """
    from_node.out_edges[to_node.task_idx] = (
        label,
        control_dependency,
    )
    to_node.in_edges[from_node.task_idx] = (
        label,
        control_dependency,
    )


def _push_candidate_node_if_ready(
    actor_to_candidates: Dict["ray._raylet.ActorID", List[_DAGOperationGraphNode]],
    node: _DAGOperationGraphNode,
) -> None:
    if node.nccl_op is not None:
        node.nccl_op.ready_task_idxs.add(node.task_idx)
    if node.is_ready:
        if node.nccl_op is None or not node.nccl_op.scheduled:
            heapq.heappush(
                actor_to_candidates[node.actor_handle._actor_id],
                node,
            )
        if node.nccl_op is not None:
            node.nccl_op.scheduled = True


def _select_next_nodes(
    actor_to_candidates: Dict["ray._raylet.ActorID", List[_DAGOperationGraphNode]],
    graph: Dict[int, _DAGOperationGraphNode],
) -> Optional[List[_DAGOperationGraphNode]]:
    """
    This function selects the next nodes for the topological sort to generate
    execution schedule. If there are multiple candidate _DAGOperationGraphNodes,
    select the node with the top priority. The priority is defined in
    `_DAGOperationGraphNode.__lt__`.

    For the implementation details, we maintain a priority queue for each actor,
    where the head of the priority queue is the node with the smallest `exec_task_idx`.

    When a node is ready, it is added to the corresponding actor's priority queue.
    For a node other than a NCCL operation node, it is ready to be executed if it
    has a zero in-degree. For a NCCL operation node, it is ready to be executed
    when all the nodes in the NCCL operation have zero in-degrees.

    If the selected node is a NCCL operation node, all the nodes in the operation
    are selected.

    Args:
        actor_to_candidates: A dictionary mapping an actor id to a list of
            candidate nodes. The list is maintained as a priority queue, so
            the head of the queue, i.e., `candidates[0]`, is the "smallest" node.
        graph: A dictionary mapping the index of a task to a dictionary of its
            _DAGOperationGraphNodes for different operations.

    Returns:
        A list of _DAGOperationGraphNodes to be placed into the corresponding
        execution schedules.
    """
    top_priority_node = None
    for candidates in actor_to_candidates.values():
        if len(candidates) == 0:
            continue
        if top_priority_node is None or candidates[0] < top_priority_node:
            top_priority_node = candidates[0]

    if top_priority_node is None:
        return None

    heapq.heappop(actor_to_candidates[top_priority_node.actor_handle._actor_id])
    if top_priority_node.nccl_op is not None:
        next_nodes = [graph[idx] for idx in top_priority_node.nccl_op.task_idxs]
    else:
        next_nodes = [top_priority_node]
    return next_nodes


def _build_dag_node_operation_graph(
    idx_to_task: Dict[int, "ray.dag.compiled_dag_node.CompiledTask"],
    actor_to_op_nodes: Dict["ray.actor.ActorHandle", List[_DAGOperationGraphNode]],
) -> Dict[int, _DAGOperationGraphNode]:
    """
    Generate a DAG node operation graph by adding edges based on the
    following rules:

    #1  Add a control edge from task with bind_index i to task with bind_index
        i+1 if they belong to the same actor.
    #2  Add data edges from a task to its downstream tasks.

    This is the step one of building an execution schedule for each actor.

    Args:
        idx_to_task: A dictionary that maps the `task_idx` to the `CompiledTask`.
            `CompiledTask` contains information about a DAGNode and its downstream
            nodes.

        actor_to_operation_nodes: A dictionary that maps an actor handle to a list
            of _DAGOperationGraphNodes. For the same actor, the index in the list
            corresponds to the index of the ExecutableTask in the list of
            `executable_tasks` in `actor_to_executable_tasks`.

    Returns:
        A graph where each node is a _DAGOperationGraphNode. The key is `task_idx`,
        the index to retrieve its task from `idx_to_task`, and the value is the
        corresponding _DAGOperationGraphNode.
    """
    # Import `ray.dag` here to avoid circular import.
    from ray.dag import ClassMethodNode, MultiOutputNode

    assert idx_to_task
    graph: Dict[int, _DAGOperationGraphNode] = {}

    for op_nodes in actor_to_op_nodes.values():
        prev_node = None
        for node in op_nodes:
            # Add an edge from task with `bind_index` i to task with
            # `bind_index` i+1 if they belong to the same actor.
            if not node.requires_nccl_op and prev_node is not None:
                _add_edge(prev_node, node, control_dependency=True)
                prev_node = node
            task_idx = node.task_idx
            assert task_idx not in graph
            graph[task_idx] = node

    # Add edges from a task to its downstream tasks.
    for task_idx, task in idx_to_task.items():
        if not (isinstance(task.dag_node, ClassMethodNode)):
            # The graph is used to generate an execution schedule for each actor.
            # The edge from the InputNode has no impact on the final execution schedule.
            continue
        if (
            isinstance(task.dag_node, ClassMethodNode)
            and task.dag_node.is_class_method_output
        ):
            continue
        for downstream_task_idx in task.downstream_task_idxs:
            downstream_dag_node = idx_to_task[downstream_task_idx].dag_node
            if isinstance(downstream_dag_node, MultiOutputNode):
                continue
            if (
                isinstance(downstream_dag_node, ClassMethodNode)
                and downstream_dag_node.is_class_method_output
            ):
                continue
            if graph[task_idx].requires_nccl_write:
                assert graph[downstream_task_idx].requires_nccl_read
            else:
                _add_edge(
                    graph[task_idx],
                    graph[downstream_task_idx],
                    "shm",
                )
                if graph[downstream_task_idx].requires_nccl_collective:
                    graph[downstream_task_idx].nccl_collective_input_idx = task_idx

    return graph


def _actor_viz_str(actor: "ray.actor.ActorHandle"):
    """
    A string representation of an actor in the visualization of the execution schedule.

    Args:
        actor: The actor to be represented.
    """
    class_name = actor._ray_actor_creation_function_descriptor.class_name
    actor_id = actor._ray_actor_id.hex()
    return f"Actor class name: {class_name}\nActor ID: {actor_id}"


def _node_viz_str(node: _DAGOperationGraphNode, idx: int, optimized_index: int):
    """
    A string representation of a node in the visualization of the execution schedule.

    Args:
        node: The node to be represented.
        idx: The index of the node in the execution schedule.
        optimized_index: The index of the node in the optimized execution schedule.
    """
    return node.viz_str() + f" {idx},{optimized_index}"


def _visualize_execution_schedule(
    actor_to_execution_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
    actor_to_overlapped_schedule: Optional[
        Dict["ray.actor.ActorHandle", List[_DAGOperationGraphNode]]
    ],
    graph: Dict[int, _DAGOperationGraphNode],
):
    """
    Visualize the execution schedule for each actor.

    The visualization will be saved as a PNG file named `compiled_graph_schedule.png`.
    Details of the visualization: # noqa

        Node description format:
            [<task_index>] <method_name> <operation> <orig_index>, <overlap_index>

        Node description fields:
            orig_index: the index in the original execution schedule
            overlap_index: the index in the overlap-communication optimized execution schedule
            If this is different from orig_index, the node is highlighted in red color

        Node grouping:
            The nodes belonging to the same actor are grouped in the same rectangle
            The actor class name and the actor id are shown in the rectangle

        Edges:
            black color (without label): data dependency
            black color (annotated with "shm"): shared memory channel
            blue color (annotated with "nccl): NCCL channel
            dashed edge: control dependency between compute operations

    Args:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the execution schedule which is a list of operation nodes.
        actor_to_overlapped_schedule: A dictionary that maps an actor handle to the
            optimized execution schedule which is a list of operation nodes.
        graph: A graph where each node is a _DAGOperationGraphNode. The key is
            `task_idx`, the index to retrieve its task from `idx_to_task`, and
            the value is the corresponding _DAGOperationGraphNode. It is generated
            by `_build_dag_node_operation_graph`.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "Please install graphviz to visualize the execution schedule. "
            "You can install it by running `pip install graphviz`."
        )

    dot = graphviz.Digraph(comment="DAG")
    node_to_viz: Dict[_DAGOperationGraphNode, str] = {}

    if actor_to_overlapped_schedule is None:
        # TODO(rui): make the visualization more concise by only displaying
        # the original schedule
        actor_to_overlapped_schedule = actor_to_execution_schedule
    for actor, execution_nodes in actor_to_execution_schedule.items():
        overlapped_schedule = actor_to_overlapped_schedule[actor]
        node_to_optimized_index = {
            node: i for i, node in enumerate(overlapped_schedule)
        }

        with dot.subgraph(name=f"cluster_{execution_nodes[0]._actor_id}") as subgraph:
            subgraph.attr(
                rank=execution_nodes[0]._actor_id, label=_actor_viz_str(actor)
            )
            for i, node in enumerate(execution_nodes):
                optimized_index = node_to_optimized_index.get(node)
                node_viz = _node_viz_str(node, i, optimized_index)
                color = "red" if optimized_index != i else "black"
                subgraph.node(node_viz, node_viz, color=color)
                node_to_viz[node] = node_viz

    for actor, execution_nodes in actor_to_execution_schedule.items():
        for i, node in enumerate(execution_nodes):
            node_viz = node_to_viz[node]
            for out_task_idx, viz_info in node.out_edges.items():
                label, control_dependency = viz_info
                out_node = graph[out_task_idx]
                out_node_repr = node_to_viz[out_node]
                color = "blue" if label == "nccl" else "black"
                style = "dashed" if control_dependency else "solid"
                dot.edge(node_viz, out_node_repr, label=label, color=color, style=style)

    # Add legend
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", labelloc="t", fontsize="20", bgcolor="lightgrey")

        # Single node and its explanation
        legend.node("example_node", "[0] bwd C 10,10\n")
        explanation = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'  # noqa
            '<TR><TD ALIGN="LEFT"><B>Node description format:</B></TD></TR>'
            '<TR><TD ALIGN="LEFT">[&lt;task_index&gt;] &lt;method_name&gt; &lt;operation&gt; &lt;orig_index&gt;, &lt;overlap_index&gt;</TD></TR>'  # noqa
            "<TR><TD></TD></TR>"
            '<TR><TD ALIGN="LEFT"><B>Node description fields:</B></TD></TR>'
            '<TR><TD ALIGN="LEFT">orig_index: the index in the original execution schedule</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT">overlap_index: the index in the overlap-communication optimized execution schedule</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT">If this is different from orig_index, the node is highlighted in <FONT COLOR="red">red color</FONT></TD></TR>'  # noqa
            "<TR><TD></TD></TR>"
            '<TR><TD ALIGN="LEFT"><B>Node grouping:</B></TD></TR>'
            '<TR><TD ALIGN="LEFT">The nodes belonging to the same actor are grouped in the same rectangle</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT">The actor class name and the actor id are shown in the rectangle</TD></TR>'  # noqa
            "<TR><TD></TD></TR>"
            '<TR><TD ALIGN="LEFT"><B>Edges:</B></TD></TR>'
            '<TR><TD ALIGN="LEFT">black color (without label): data dependency</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT">black color (annotated with "shm"): shared memory channel</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT"><FONT COLOR="blue">blue color</FONT> (annotated with "nccl"): NCCL channel</TD></TR>'  # noqa
            '<TR><TD ALIGN="LEFT">dashed edge: control dependency between compute operations</TD></TR>'  # noqa
            "</TABLE>>"
        )

        legend.node("example_explanation", explanation, shape="plaintext")
        legend.edge("example_node", "example_explanation", style="invis")

    logger.info(
        "Writing compiled graph schedule visualization "
        "to compiled_graph_schedule.png"
    )
    dot.render("compiled_graph_schedule", format="png", view=False)


def _generate_actor_to_execution_schedule(
    graph: Dict[int, _DAGOperationGraphNode]
) -> Dict["ray.actor.ActorHandle", List[_DAGOperationGraphNode]]:
    """
    Generate an execution schedule for each actor. The schedule is a list of
    operation nodes to be executed. The function uses a topological sort
    algorithm to generate the schedule.

    Args:
        graph: A graph where each node is a _DAGOperationGraphNode. The key is
            `task_idx`, the index to retrieve its task from `idx_to_task`, and
            the value is the corresponding _DAGOperationGraphNode. It is generated
            by `_build_dag_node_operation_graph`.

    Returns:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the execution schedule which is a list of operation nodes to be
            executed.
    """

    # Mapping from the actor handle to the execution schedule which is a list
    # of operations to be executed.
    actor_to_execution_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ] = defaultdict(list)

    # A dictionary mapping an actor id to a list of candidate nodes. The list
    # is maintained as a priority queue, so the head of the queue, i.e.,
    # `candidates[0]`, is the "smallest" node.
    actor_to_candidates: Dict["ray._raylet.ActorID", List[_DAGOperationGraphNode]] = (
        defaultdict(list)
    )
    for node in graph.values():
        if node.in_degree == 0:
            _push_candidate_node_if_ready(actor_to_candidates, node)

    visited_nodes: Set[_DAGOperationGraphNode] = set()

    # Use topological sort algorithm to generate the execution schedule.
    while True:
        # Select a list of nodes to be executed. There are two cases:
        # 1. If the selected node is not a NCCL operation, only itself is returned.
        # 2. If the selected node is a NCCL operation, all the nodes in the operation
        #    are returned.
        nodes = _select_next_nodes(actor_to_candidates, graph)
        if nodes is None:
            break
        # Add the selected nodes to the execution schedule.
        for node in nodes:
            assert node not in visited_nodes
            actor_to_execution_schedule[node.actor_handle].append(node)
            visited_nodes.add(node)
        # Update the in-degree of the downstream nodes.
        for node in nodes:
            for out_node_task_idx in node.out_edges:
                out_node = graph[out_node_task_idx]
                out_node.in_edges.pop(node.task_idx)
                if out_node.in_degree == 0:
                    _push_candidate_node_if_ready(actor_to_candidates, out_node)
    assert len(visited_nodes) == len(graph), "Expected all nodes to be visited"
    for node in visited_nodes:
        assert node.is_ready, f"Expected {node} to be ready"
    for candidates in actor_to_candidates.values():
        assert len(candidates) == 0, "Expected all candidates to be empty"

    return actor_to_execution_schedule


def _generate_overlapped_execution_schedule(
    actor_to_execution_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ],
) -> Dict["ray.actor.ActorHandle", List[_DAGOperationGraphNode]]:
    """
    From an existing execution schedule, generate a new schedule by overlapping
    computation and communication.

    Currently, the algorithm generates a new schedule for each actor as follows:
    For each NCCL read operation (i.e., recv), scan backwards to find the nearest
    compute node to swap with so that the NCCL read operation can be overlapped
    with computation.

    [CL]
    Collective operations are not yet supported.

    Args:
        actor_to_execution_schedule: A dictionary that maps an actor handle to
            the existing execution schedule for the actor. The schedule is a list
            of operations to be executed.

    Returns:
        A dictionary that maps an actor handle to the overlapped execution schedule
        for the actor.
    """

    actor_to_overlapped_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ] = copy.deepcopy(actor_to_execution_schedule)
    """ [NOTE] Comment out for manual DDP overlap.
    for overlapped_schedule in actor_to_overlapped_schedule.values():
        # Swap each NCCL read with the previous compute to overlap the NCCL read
        # with computation.
        for i in range(1, len(overlapped_schedule)):
            if not overlapped_schedule[i - 1].requires_nccl_op and (
                overlapped_schedule[i].requires_nccl_read
                or (
                    overlapped_schedule[i].requires_nccl_collective
                    and overlapped_schedule[i].nccl_collective_input_idx
                    != overlapped_schedule[i - 1].task_idx
                )
            ):
                overlapped_schedule[i], overlapped_schedule[i - 1] = (
                    overlapped_schedule[i - 1],
                    overlapped_schedule[i],
                )
    """
    return actor_to_overlapped_schedule


def _extract_execution_schedule(
    actor_to_execution_schedule: Dict[
        "ray.actor.ActorHandle", List[_DAGOperationGraphNode]
    ]
) -> Dict["ray.actor.ActorHandle", List[_DAGNodeOperation]]:
    """
    Extract _DAGNodeOperation from _DAGOperationGraphNode in the schedule
    and discard unnecessary information.
    """
    return {
        actor: [node.op for node in nodes]
        for actor, nodes in actor_to_execution_schedule.items()
    }
