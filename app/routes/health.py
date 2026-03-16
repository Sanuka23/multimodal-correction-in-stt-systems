"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    model_loaded = False
    backend = "unknown"
    try:
        from asr_correction.model import _model
        model_loaded = _model is not None
    except Exception:
        pass

    try:
        import platform
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            backend = "mlx"
        else:
            backend = "transformers"
    except Exception:
        pass

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "backend": backend,
        "version": "1.0.0",
    }
