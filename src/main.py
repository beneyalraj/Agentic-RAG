import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI starting up...")
    yield
    logger.info("FastAPI shutting down...")

app = FastAPI(
    title="Agentic RAG Assistant",
    description="Financial agent over JPMorgan Chase SEC filings",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["agent"])

@app.get("/")
async def root():
    return {"message": "Agentic RAG Assistant API is running. Use POST /api/v1/query."}