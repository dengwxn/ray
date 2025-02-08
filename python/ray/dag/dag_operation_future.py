from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, TypeVar
from ray.util.annotations import DeveloperAPI
# [NOTE:print]
import torch


if TYPE_CHECKING:
    import cupy as cp

T = TypeVar("T")


@DeveloperAPI
class DAGOperationFuture(ABC, Generic[T]):
    """
    A future representing the result of a DAG operation.

    This is an abstraction that is internal to each actor,
    and is not exposed to the DAG caller.
    """

    # [NOTE:print]
    @abstractmethod
    def wait(self, method_name: Optional[str] = None):
        """
        Wait for the future and return the result of the operation.
        """
        raise NotImplementedError


@DeveloperAPI
class ResolvedFuture(DAGOperationFuture):
    """
    A future that is already resolved. Calling `wait()` on this will
    immediately return the result without blocking.
    """

    def __init__(self, result):
        """
        Initialize a resolved future.

        Args:
            result: The result of the future.
        """
        self._result = result

    # [NOTE:print]
    def wait(self, method_name: Optional[str] = None):
        """
        Wait and immediately return the result. This operation will not block.
        """
        return self._result


@DeveloperAPI
class GPUFuture(DAGOperationFuture[Any]):
    """
    A future for a GPU event on a CUDA stream.

    This future wraps a buffer, and records an event on the given stream
    when it is created. When the future is waited on, it makes the current
    CUDA stream wait on the event, then returns the buffer.

    The buffer must be a GPU tensor produced by an earlier operation launched
    on the given stream, or it could be CPU data. Then the future guarantees
    that when the wait() returns, the buffer is ready on the current stream.

    The `wait()` does not block CPU.
    """

    # [HACK] This prevents CUDA events from being garbage collected.
    # id: int = 0
    # id_to_event: Dict[int, "cp.cuda.Event"] = {}

    def __init__(self, buf: Any, stream: Optional["cp.cuda.Stream"] = None):
        """
        Initialize a GPU future on the given stream.

        Args:
            buf: The buffer to return when the future is resolved.
            stream: The CUDA stream to record the event on, this event is waited
                on when the future is resolved. If None, the current stream is used.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._buf = buf
        self._event = cp.cuda.Event()
        self._event.record(stream)
        # [HACK]
        # self._id = GPUFuture.id
        # GPUFuture.id += 1
        # GPUFuture.id_to_event[self._id] = self._event

    # [NOTE:print]
    def wait(self, method_name: Optional[str] = None) -> Any:
        """
        Wait for the future on the current CUDA stream and return the result from
        the GPU operation. This operation does not block CPU.
        """
        import cupy as cp

        current_stream = cp.cuda.get_current_stream()
        current_stream.wait_event(self._event)
        # [HACK]
        # GPUFuture.id_to_event.pop(self._id)
        # [NOTE:print]
        if isinstance(self._buf, torch.Tensor):
            data_ptr = self._buf.data_ptr()
        else:
            data_ptr = None
        if method_name is not None:
            print(f"[{method_name}.wait] {self._buf}@{id(self._buf)}[{data_ptr}]")
        return self._buf
