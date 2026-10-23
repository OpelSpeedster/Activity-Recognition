"""Select a practical YOLO pose profile based on the requested device."""

from config import (
    YOLO_POSE_IMGSZ_CPU,
    YOLO_POSE_IMGSZ_CUDA,
    YOLO_POSE_MODEL,
    YOLO_POSE_MODEL_CPU,
    YOLO_POSE_MODEL_CUDA,
)


def resolve_pose_profile(requested_device=None, model_override=None, imgsz_override=None):
    """Return model, device, and image size, falling back to CPU if needed."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if requested_device is None or str(requested_device).lower() == "auto":
        use_cuda = cuda_available
        device = "cuda:0" if use_cuda else "cpu"
    else:
        requested = str(requested_device).lower()
        wants_cuda = requested.startswith("cuda") or requested.isdigit()
        if wants_cuda and not cuda_available:
            print("[WARN] CUDA was requested but is unavailable; using CPU.")
            use_cuda = False
            device = "cpu"
        else:
            use_cuda = wants_cuda
            device = "cuda:0" if requested == "cuda" else requested_device

    model = model_override or YOLO_POSE_MODEL or (
        YOLO_POSE_MODEL_CUDA if use_cuda else YOLO_POSE_MODEL_CPU
    )
    imgsz = imgsz_override or (YOLO_POSE_IMGSZ_CUDA if use_cuda else YOLO_POSE_IMGSZ_CPU)
    return model, device, imgsz
