from fastapi import APIRouter

from src.config import settings

router = APIRouter()


@router.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "engine": settings.engine_type.value}
