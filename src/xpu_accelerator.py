from typing import Union

import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.exceptions import MisconfigurationException


def _parse_xpu_devices(devices: Union[int, str]) -> int:
    if devices == "auto":
        return 1
    if isinstance(devices, str):
        try:
            devices = int(devices)
        except ValueError as exc:
            raise MisconfigurationException(
                f"XPU accelerator expects a single integer device count, got {devices!r}."
            ) from exc
    if devices != 1:
        raise MisconfigurationException(
            f"XPU accelerator currently supports exactly one device, got {devices}."
        )
    return devices


class XPUAccelerator(Accelerator):
    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device} instead.")
        torch.xpu.set_device(device)

    def get_device_stats(self, device: torch.device) -> dict[str, object]:
        stats = torch.xpu.memory_stats(device)
        return {key: value for key, value in stats.items() if isinstance(value, (int, float))}

    def teardown(self) -> None:
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str]) -> int:
        return _parse_xpu_devices(devices)

    @staticmethod
    def get_parallel_devices(devices: Union[int, str]) -> list[torch.device]:
        _parse_xpu_devices(devices)
        return [torch.device("xpu", 0)]

    @staticmethod
    def auto_device_count() -> int:
        return 1

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        return "xpu"
