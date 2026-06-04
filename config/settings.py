from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL       = os.environ["OLLAMA_BASE_URL"]
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY        = os.getenv("OLLAMA_API_KEY", "")

MODEL_STRATEGIST  = os.getenv("MODEL_STRATEGIST", "deepseek-v4-pro")
MODEL_REFLECTOR   = os.getenv("MODEL_REFLECTOR",  "glm-5")
MODEL_CODER       = os.getenv("MODEL_CODER",       "qwen3-coder-next")
MODEL_EMBEDDING   = os.getenv("MODEL_EMBEDDING",   "qwen3-embedding:8b")

ACTION_TYPES: list[str] = [
    "feature_engineering",
    "model_swap",
    "hyperparam_search",
    "preprocessing",
    "ensemble",
]
