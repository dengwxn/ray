import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

from ray.experimental.channel import ChannelContext
from ray.experimental.channel.common import ChannelInterface
from ray.experimental.channel.nccl_group import _NcclGroup
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    import torch

    from ray.experimental.channel.shared_memory_channel import Channel
    from ray.experimental.channel.torch_tensor_type import TorchTensorType


# Logger for this module. It should be configured at the entry point
# into the program using Ray. Ray provides a default configuration at
# entry/init points.
logger = logging.getLogger(__name__)


# Signature for a torch.Tensor allocator is:
# (shape: Tuple[int], dtype: torch.dtype) -> torch.Tensor.
TorchTensorAllocator = Callable[[Tuple[int], "torch.dtype"], "torch.Tensor"]


def _torch_zeros_allocator(shape: Tuple[int], dtype: "torch.dtype"):
    import torch

    ctx = ChannelContext.get_current()
    return torch.zeros(shape, dtype=dtype, device=ctx.torch_device)


@DeveloperAPI
class TorchTensorNcclCollectiveChannel(ChannelInterface):
    def __init__(
        self,
        typ: "TorchTensorType",
    ):
        """
        Create a channel for torch.Tensors transferred via NCCL.

        Args:
            typ: Type information about the values passed through the channel.
        """
        import torch

        from ray.experimental.channel.torch_tensor_type import TorchTensorType

        self.torch: ModuleType = torch

        assert isinstance(typ, TorchTensorType)
        assert typ.transport == typ.NCCL_ALLREDUCE
        self._typ: "TorchTensorType" = typ

        ctx = ChannelContext.get_current()
        assert self._typ.nccl_group_id is not None, "No NCCL group specified."
        self._nccl_group_id: str = self._typ.nccl_group_id
        self._nccl_group: "_NcclGroup" = ctx.nccl_groups[self._typ.nccl_group_id]
        assert (
            self._nccl_group is not None
        ), "ChannelContext.nccl_group is not initialized."

    def __reduce__(self):
        return (
            self.__class__,
            (self._typ,),
        )

    def ensure_registered_as_writer(self):
        """
        Check whether the process is a valid writer. This method must be idempotent.
        """
        pass

    def ensure_registered_as_reader(self):
        """
        Check whether the process is a valid reader. This method must be idempotent.
        """
        pass

    def write(self, value: Any, timeout: Optional[float] = None) -> None:
        """
        Write a value to the channel.

        Blocks if there are still pending readers for the previous value. The
        writer may not write again until the specified number of readers have
        read the value.

        Args:
            value: The value to write.
            timeout: The maximum time in seconds to wait to write the value.
                None means using default timeout, 0 means immediate timeout
                (immediate success or timeout without blocking), -1 means
                infinite timeout (block indefinitely).
        """
        pass

    """[TODO] Simplify.
    def _read_single_tensor(self, typ: "TorchTensorType") -> "torch.Tensor":
        buf = self._torch_tensor_allocator(typ._shape, typ._dtype)
        self._nccl_group.recv(buf, self._writer_rank)
        return buf
    """

    def read(
        self, timeout: Optional[float] = None
    ) -> Union["torch.Tensor", List["torch.Tensor"]]:
        # [TODO] sync before read
        """[TODO] Simplify.
        if self._meta_channel is not None:
            meta = self._meta_channel.read()
        else:
            meta = self._typ

        if not isinstance(meta, list):
            return self._read_single_tensor(meta)
        """

        bufs: List["torch.Tensor"] = []
        """[TODO] Simplify.
        for typ in meta:
            bufs.append(self._read_single_tensor(typ))
        """
        # TODO: Sync CUDA stream after receiving all tensors, instead of after
        # each tensor.
        return bufs

    def close(self) -> None:
        self._nccl_group.destroy()
        ctx = ChannelContext.get_current()
        if self._nccl_group_id in ctx.nccl_groups:
            del ctx.nccl_groups[self._nccl_group_id]

    def has_static_type(self) -> bool:
        from ray.experimental.channel.torch_tensor_type import TorchTensorType

        return (
            self._typ._shape != TorchTensorType.AUTO
            and self._typ._dtype != TorchTensorType.AUTO
        )
