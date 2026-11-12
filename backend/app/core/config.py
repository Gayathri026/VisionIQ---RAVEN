import os
from typing import Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "VisionIQ: Adaptive Multi-Agent Vision Intelligence"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite:///./visioniq.db"

    # Monorepo Storage Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    STORAGE_DIR: str = "storage"
    UPLOADS_DIR: str = "storage/uploads"
    REPORTS_DIR: str = "storage/reports"
    ANNOTATIONS_DIR: str = "storage/annotations"
    KNOWLEDGE_BASE_DIR: str = "knowledge_base"

    # Upload Validation Settings
    MAX_IMAGE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    # Model Registries & Checkpoint Paths
    MODEL_YOLO_PATH: str = "visioniq/backend/app/models/yolo11n.pt"
    MODEL_PPE_PATH: str = "visioniq/backend/app/models/yolo_ppe.pt"
    MODEL_DEFECT_PATH: str = "visioniq/backend/app/models/defect_model.pt"

    # LLM Settings (optional - deterministic fallback provided)
    OPENAI_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o"

    # Research Routing Parameters (Phase-I calibrated weights)
    # S(a, q) = alpha * R(a,q) + delta * E(a) - beta * C(a) - gamma * L(a)
    ROUTING_ALPHA: float = Field(0.50, description="Query relevance weight")
    ROUTING_BETA: float = Field(0.20, description="Computational cost penalty weight")
    ROUTING_GAMMA: float = Field(0.15, description="Latency penalty weight")
    ROUTING_DELTA: float = Field(0.15, description="Expected evidence capability weight")
    ROUTING_THRESHOLD: float = Field(0.45, description="Calibrated operating threshold tau")

    # Phase 2: Dynamic Cost and Latency Feedback
    ENABLE_DYNAMIC_LATENCY_FEEDBACK: bool = Field(
        False, 
        description="Whether to use historical EMA measured latency feedback in Equation (1). Static prior used when False."
    )
    ROUTING_LATENCY_WINDOW_SIZE: int = Field(
        50, 
        description="Rolling window size W for historical latency observations per agent"
    )
    ROUTING_LATENCY_EMA_ALPHA: float = Field(
        0.30, 
        description="Exponential Moving Average (EMA) smoothing factor alpha in (0.0, 1.0]"
    )
    ROUTING_LATENCY_MIN_OBSERVATIONS: int = Field(
        3, 
        description="Minimum number of valid historical observations required before activating EMA"
    )
    ROUTING_LATENCY_NORMALIZATION_METHOD: str = Field(
        "anchored", 
        description="Latency normalization method: 'anchored' (proportional to benchmark prior) or 'linear' (scale-max)"
    )
    ROUTING_LATENCY_MAX_SCALE_MS: float = Field(
        5000.0, 
        description="Reference maximum execution latency in milliseconds for linear normalization"
    )
    ROUTING_BENCHMARK_LATENCIES_MS: Dict[str, float] = {
        "intent": 5.0,
        "object": 250.0,
        "ppe": 100.0,
        "ocr": 3000.0,
        "defect": 50.0,
        "scene": 20.0,
        "evidence_validation": 2.0,
        "rag": 700.0,
        "rca": 2.0,
        "recommendation": 2.0,
        "report": 15.0
    }
    HARDWARE_DEVICE_CONTEXT: str = Field(
        "cpu", 
        description="Current hardware device execution context ('cpu' or 'cuda')"
    )
    AGENT_MODEL_VERSIONS: Dict[str, str] = {
        "intent": "VisionIQ-IntentClassifier-v1.0",
        "object": "YOLOv11n (Ultralytics 8.3.0)",
        "ppe": "VisionIQ-PPE-SpatialAssoc-v1.0",
        "ocr": "EasyOCR-1.7.2",
        "defect": "IndustrialDefectNet-v1.0 (PyTorch CNN)",
        "scene": "VisionIQ-SceneContext-v1.0",
        "evidence_validation": "VisionIQ-ProvenanceValidator-v1.0",
        "rag": "VisionIQ-StandardsRAG-v1.0",
        "rca": "VisionIQ-CausalReasoner-v1.0",
        "recommendation": "VisionIQ-ActionPlanner-v1.0",
        "report": "VisionIQ-PDF-Engine-v1.0"
    }

    @field_validator("ROUTING_ALPHA", "ROUTING_BETA", "ROUTING_GAMMA", "ROUTING_DELTA", "ROUTING_THRESHOLD")
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be normalized in [0.0, 1.0], got {v}")
        return v

    @field_validator("ROUTING_LATENCY_EMA_ALPHA")
    @classmethod
    def validate_ema_alpha(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"ROUTING_LATENCY_EMA_ALPHA must be in (0.0, 1.0], got {v}")
        return v

    # Per-Agent baseline normalized costs C(a) in [0, 1]
    AGENT_COSTS: Dict[str, float] = {
        "intent": 0.05,
        "object": 0.15,
        "ppe": 0.20,
        "ocr": 0.30,
        "defect": 0.35,
        "scene": 0.10,
        "evidence_validation": 0.10,
        "rag": 0.25,
        "rca": 0.20,
        "recommendation": 0.15,
        "report": 0.10
    }

    # Per-Agent baseline normalized expected latencies L(a) in [0, 1]
    AGENT_LATENCIES: Dict[str, float] = {
        "intent": 0.05,
        "object": 0.18,
        "ppe": 0.22,
        "ocr": 0.40,
        "defect": 0.35,
        "scene": 0.12,
        "evidence_validation": 0.08,
        "rag": 0.25,
        "rca": 0.20,
        "recommendation": 0.15,
        "report": 0.15
    }

    # Per-Agent expected evidence capability E(a) in [0, 1]
    AGENT_CAPABILITIES: Dict[str, float] = {
        "object": 0.85,
        "ppe": 0.90,
        "ocr": 0.80,
        "defect": 0.85,
        "scene": 0.75,
        "evidence_validation": 0.90,
        "rag": 0.85,
        "rca": 0.80,
        "recommendation": 0.85,
        "report": 0.95
    }

    MAX_IMAGE_SIZE_MB: int = 15
    MAX_VIDEO_SIZE_MB: int = 100
    DEFAULT_VIDEO_SAMPLE_FPS: float = 1.0

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure required directories exist
for p in [settings.STORAGE_DIR, settings.UPLOADS_DIR, settings.REPORTS_DIR, 
          settings.ANNOTATIONS_DIR, settings.KNOWLEDGE_BASE_DIR]:
    os.makedirs(p, exist_ok=True)
