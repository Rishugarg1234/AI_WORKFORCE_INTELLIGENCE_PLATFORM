"""
Model loader module for loading the trained pipeline and metadata safely.
"""

import json
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from app.utils.config import MODEL_PIPELINE_PATH, MODEL_METADATA_PATH
from app.utils.logger import logger

class ModelRegistry:
    """Thread-safe lazy-loading singleton for ML artifacts."""
    _instance: Optional["ModelRegistry"] = None
    _pipeline: Optional[Any] = None
    _metadata: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Loads and returns the trained pipeline and metadata."""
        if self._pipeline is None or self._metadata is None:
            if not MODEL_PIPELINE_PATH.exists():
                logger.error(f"Trained model artifact not found at {MODEL_PIPELINE_PATH}")
                raise FileNotFoundError(f"Model file missing: {MODEL_PIPELINE_PATH}")
                
            if not MODEL_METADATA_PATH.exists():
                logger.error(f"Model metadata not found at {MODEL_METADATA_PATH}")
                raise FileNotFoundError(f"Metadata file missing: {MODEL_METADATA_PATH}")

            logger.info(f"Loading attrition model pipeline from {MODEL_PIPELINE_PATH}")
            self._pipeline = joblib.load(MODEL_PIPELINE_PATH)
            
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
                
            logger.info(f"Loaded model version {self._metadata.get('version', 'unknown')} ({self._metadata.get('algorithm', 'unknown')})")

        return self._pipeline, self._metadata

model_registry = ModelRegistry()

def get_attrition_pipeline() -> Any:
    """Convenience getter for pipeline."""
    pipeline, _ = model_registry.load_model()
    return pipeline

def get_model_metadata() -> Dict[str, Any]:
    """Convenience getter for metadata."""
    _, metadata = model_registry.load_model()
    return metadata
