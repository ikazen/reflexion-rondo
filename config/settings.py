from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL   = os.environ["OLLAMA_BASE_URL"]
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY", "")

MODEL_STRATEGIST  = os.getenv("MODEL_STRATEGIST", "qwen3.5:14b")
MODEL_REFLECTOR   = os.getenv("MODEL_REFLECTOR",  "qwen3.5:14b")
MODEL_CODER       = os.getenv("MODEL_CODER",       "devstral-small-2")
MODEL_EMBEDDING   = os.getenv("MODEL_EMBEDDING",   "qwen3-embedding:8b")
