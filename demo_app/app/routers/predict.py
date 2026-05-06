import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.inference.pipeline import run_pipeline

router = APIRouter()
_lock = asyncio.Lock()  # serialise GPU inference


@router.post(
    "/predict",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict(video: UploadFile = File(...)):
    filename = video.filename or ""
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Only .mp4 files are accepted")

    async with _lock:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
        try:
            loop = asyncio.get_event_loop()
            png_bytes = await loop.run_in_executor(None, run_pipeline, tmp_path)
        finally:
            os.unlink(tmp_path)

    return Response(content=png_bytes, media_type="image/png")
