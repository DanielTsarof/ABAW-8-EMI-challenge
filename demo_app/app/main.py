from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import predict

app = FastAPI(title="EMI Emotion Demo")
app.include_router(predict.router, prefix="/api")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")
