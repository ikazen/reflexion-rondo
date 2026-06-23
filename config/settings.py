from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY        = os.getenv("OLLAMA_API_KEY", "")

MODEL_STRATEGIST  = os.getenv("MODEL_STRATEGIST",  "deepseek-v4-pro")
MODEL_REFLECTOR   = os.getenv("MODEL_REFLECTOR",   "kimi-k2.6")
MODEL_CODER       = os.getenv("MODEL_CODER",        "qwen3-coder-next")
MODEL_EMBEDDING   = os.getenv("MODEL_EMBEDDING",    "qwen3-embedding:8b")

ACTION_TYPES: list[str] = [
    "feature_engineering",
    "model_swap",
    "hyperparam_search",
    "preprocessing",
    "ensemble",
]

LABEL_Z: float = 1.0

# 승격 cross-seed 확인: 이 seed 목록 전부에서 gain_vs_best > 0 재현돼야 승격
PROMOTE_CONFIRM_SEEDS: list[int] = [
    int(s) for s in os.getenv("PROMOTE_CONFIRM_SEEDS", "7,101").split(",")
]

_CLASSIFICATION_TASK_TYPES = frozenset({"binary", "multiclass"})


def is_classification(task_type: str) -> bool:
    """Derive is_classification from task_type. Single source of truth."""
    return task_type in _CLASSIFICATION_TASK_TYPES


def require_llm_env() -> None:
    """Validate required LLM env vars at entrypoint startup. Raises RuntimeError if missing."""
    missing = [k for k in ("OLLAMA_BASE_URL", "OLLAMA_CLOUD_BASE_URL") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required LLM env vars: {missing}")
