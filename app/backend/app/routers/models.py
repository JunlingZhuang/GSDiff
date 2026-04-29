"""Model status API endpoints."""

from fastapi import APIRouter

from app.services.model_status import model_status

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/status")
def status():
    return model_status.snapshot()
