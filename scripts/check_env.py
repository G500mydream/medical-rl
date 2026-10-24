"""Print the hardware/software facts that matter for this project."""
from __future__ import annotations

import importlib
import shutil

import torch


def main() -> None:
    print(f"torch       : {torch.__version__}")
    print(f"cuda build  : {torch.version.cuda}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    print(f"gpu count   : {torch.cuda.device_count()}")
    print(f"nvcc        : {shutil.which('nvcc') or 'not found'}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(
                f"  gpu {i}: {props.name} | {total / 2**30:.1f} GB total "
                f"| {free / 2**30:.1f} GB free | sm{props.major}{props.minor}"
            )
        try:
            x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            print(f"bf16        : ok ({y.dtype})")
        except Exception as exc:  # noqa: BLE001
            print(f"bf16        : failed ({exc})")

    for module in ("vllm", "flash_attn", "peft", "transformers", "datasets", "verl", "llamafactory"):
        try:
            mod = importlib.import_module(module)
            print(f"{module:<12}: {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{module:<12}: MISSING ({exc.__class__.__name__}: {exc})")


if __name__ == "__main__":
    main()
