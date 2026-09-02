from __future__ import annotations


def select_torch_device(torch_module, requested: str):
    if requested == "auto":
        if torch_module.cuda.is_available():
            requested = "cuda"
        elif getattr(torch_module.backends, "mps", None) is not None and (
            torch_module.backends.mps.is_available()
        ):
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch_module.device(requested)
