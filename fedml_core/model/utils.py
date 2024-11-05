from typing import Callable, Type, Union

from torch import nn

TorchModuleType = Union[str, Callable[..., nn.Module]]


def make_nn_module(module_type: TorchModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        try:
            cls = getattr(nn, module_type)
        except AttributeError as err:
            raise ValueError(
                f"Failed to construct the module {module_type} with the arguments"
                f" {args}"
            ) from err
        return cls(*args)
    else:
        return module_type(*args)