"""FastAPI application entry-point for GSDiff floorplan generation."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Log to console AND to backend.log next to this package, so full tracebacks are
# recoverable even when stdout is hidden (e.g. under `npm run dev:all`, where the
# backend stream is interleaved with the Next.js dev server).
_LOG_FILE = Path(__file__).resolve().parent.parent / "backend.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)

from app.services.model_manager import manager
from app.routers.generate import router as generate_router
from app.routers.models import router as models_router
from app.routers.retrieve import router as retrieve_router
from app.routers.procedural import router as procedural_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    manager.load_all()
    print("All models loaded!")
    yield


app = FastAPI(title="GSDiff API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)
app.include_router(models_router)
app.include_router(retrieve_router)
app.include_router(procedural_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/doc", include_in_schema=False)
def doc_alias():
    return RedirectResponse(url="/docs")
