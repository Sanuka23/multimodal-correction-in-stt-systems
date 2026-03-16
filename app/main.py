"""FastAPI application for Multimodal ASR Correction."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .database import connect_db, close_db
from .routes import correction, evaluation, training, dashboard, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Multimodal ASR Correction API on port %s", settings.api_port)
    await connect_db()
    yield
    await close_db()
    logger.info("Shutting down")


app = FastAPI(
    title="Multimodal ASR Correction API",
    version="1.0.0",
    description="Transcript correction using fine-tuned LLM with vocabulary and OCR context",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# Routes
app.include_router(health.router)
app.include_router(correction.router)
app.include_router(evaluation.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(dashboard.router)
