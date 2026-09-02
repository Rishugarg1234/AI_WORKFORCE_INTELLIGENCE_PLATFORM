"""
FastAPI Main Application Entrypoint.
Exposes endpoints for prediction, workforce intelligence, skill gaps, and health checks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime
from app.utils.config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.utils.logger import logger
from app.ml.model_loader import get_model_metadata
from app.api.attrition import router as attrition_router
from app.api.dashboard import router as dashboard_router
from app.api.skills import router as skills_router

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Enterprise HR AI API server...")
    try:
        meta = get_model_metadata()
        logger.info(f"Initialized with model: {meta.get('model_name')} v{meta.get('version')} ({meta.get('algorithm')})")
    except Exception as e:
        logger.warning(f"Model metadata warmup notice: {e}")
    logger.info("FastAPI ready to receive requests.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Enterprise HR AI API server...")

@app.get("/health", tags=["System Health"])
async def health_check():
    """Health check endpoint for monitoring and orchestration."""
    try:
        meta = get_model_metadata()
        model_ver = meta.get("version", "v1")
        model_algo = meta.get("algorithm", "XGBoost")
    except Exception:
        model_ver = "unavailable"
        model_algo = "unavailable"

    return {
        "status": "healthy",
        "service": API_TITLE,
        "version": API_VERSION,
        "model_version": model_ver,
        "model_algorithm": model_algo,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Mount Routers
app.include_router(attrition_router)
app.include_router(dashboard_router)
app.include_router(skills_router)

if __name__ == "__main__":
    import uvicorn
    from app.utils.config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
