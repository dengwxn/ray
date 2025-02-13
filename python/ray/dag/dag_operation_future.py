from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, TypeVar
from ray.util.annotations import DeveloperAPI
import time


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

    @abstractmethod
    def wait(self):
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

    def wait(self, _=None):
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
    id: int = 0
    # id_to_event: Dict[int, "cp.cuda.Event"] = {}

    def __init__(
        self,
        buf: Any,
        stream: Optional["cp.cuda.Stream"] = None,
        method_name: Optional[str] = None,
    ):
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
        self.method_name = method_name
        self.ready = False
        # [HACK]
        self._id = GPUFuture.id
        GPUFuture.id += 1
        # GPUFuture.id_to_event[self._id] = self._event
        print(
            f"[{time.perf_counter()}][{method_name}] init gpu future: {self} id={self._id} ({self._event=})"
        )
        self.fut_id = None
        # self.cache()

    def wait(self, method_name: Optional[str] = None) -> Any:
        """
        Wait for the future on the current CUDA stream and return the result from
        the GPU operation. This operation does not block CPU.
        """
        print(
            f"[{time.perf_counter()}][{method_name}] wait gpu future {self} id={self._id}"
        )
        if self.ready:
            return self._buf

        import cupy as cp

        current_stream = cp.cuda.get_current_stream()
        current_stream.wait_event(self._event)
        # del self._event
        # # [HACK]
        # GPUFuture.id_to_event.pop(self._id)
        self.ready = True

        from ray.experimental.channel.common import ChannelContext

        ctx = ChannelContext.get_current().serialization_context
        ctx.reset_gpu_future(self.fut_id)
        return self._buf

    def cache(self, fut_id: int) -> None:
        from ray.experimental.channel.common import ChannelContext

        ctx = ChannelContext.get_current().serialization_context
        ctx.set_gpu_future(fut_id, self)
        self.fut_id = fut_id

    def destroy_event(self) -> None:
        if self._event is None:
            return

        print(f"[{time.perf_counter()}] destroy event")
        import cupy as cp

        print(self._event.ptr)
        cp.cuda.runtime.eventDestroy(self._event.ptr)
        self._event.ptr = 0
        print(self._event.ptr)
        self._event = None

    # def __del__(self) -> None:
    #     print(f"[{time.perf_counter()}] del gpu future {self} id={self._id}")
    #     if not self.ready:
    #         # raise ValueError(f"{self.method_name} del w/o waiting")
    #         print(f"waiting in delete")
    #         self.wait()
    #     self._event = None
    #     self._buf = None