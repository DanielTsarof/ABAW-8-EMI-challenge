import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.inference.pipeline import run_pipeline, run_pipeline_json

router = APIRouter()
_lock = asyncio.Lock()  # serialise GPU inference


async def _save_upload(video: UploadFile) -> str:
    filename = video.filename or ""
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Only .mp4 files are accepted")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        return tmp.name


@router.post(
    "/predict",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict(video: UploadFile = File(...)):
    async with _lock:
        tmp_path = await _save_upload(video)
        try:
            loop = asyncio.get_event_loop()
            png_bytes = await loop.run_in_executor(None, run_pipeline, tmp_path)
        finally:
            os.unlink(tmp_path)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/predict/json")
async def predict_json(video: UploadFile = File(...)):
    async with _lock:
        tmp_path = await _save_upload(video)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_pipeline_json, tmp_path)
        finally:
            os.unlink(tmp_path)
    return JSONResponse(content=result)
