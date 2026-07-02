from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY        = os.getenv("OLLAMA_API_KEY", "")

MODEL_STRATEGIST  = os.getenv("MODEL_STRATEGIST",  "glm-5.2")
MODEL_REFLECTOR   = os.getenv("MODEL_REFLECTOR",   "kimi-k2.6")
MODEL_CODER       = os.getenv("MODEL_CODER",        "gpt-oss:120b")
MODEL_EMBEDDING   = os.getenv("MODEL_EMBEDDING",    "qwen3-embedding:8b")

# BON-240: qwen3.5:397b는 동일 프롬프트에서 qwen3-coder-next 대비 출력 토큰 9배(reasoning
# 모델이라 긴 thinking을 뿜는데 _extract_code가 ```python 블록만 취해 전량 버려짐). 폐기
# 공지로 코더 전문 라인(qwen3-coder-next/480b, devstral 계열)이 전부 없어져 gpt-oss:120b로
# 대체 — reasoning_effort로 토큰 예산 조절 가능한 생존 모델 중 벤치 결과가 가장 나음.
MODEL_CODER_REASONING_EFFORT = os.getenv("MODEL_CODER_REASONING_EFFORT", "medium")

# BON-193: Actor(Strategist/Coder/Reflector)는 확률적이라 attempt 간 CV 변화가
# 교훈 효과인지 LLM 샘플링 운인지 구분이 안 됐다. temperature를 명시 고정해 최소한
# 비결정성 자체를 문서화·재현 가능하게 한다. seed는 기본 미고정(탐색성 유지) —
# LLM_SEED env로 실험 시에만 고정.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_SEED: int | None = int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else None


def llm_options(**extra) -> dict:
    """Strategist/Coder/Reflector 공용 .chat() options. temperature/seed를 단일 소스로 유지."""
    opts: dict = {"temperature": LLM_TEMPERATURE}
    if LLM_SEED is not None:
        opts["seed"] = LLM_SEED
    opts.update(extra)
    return opts

ACTION_TYPES: list[str] = [
    "feature_engineering",
    "model_swap",
    "hyperparam_search",
    "preprocessing",
    "ensemble",
]

# BON-194: 1σ는 통계적으로 유의하지 않아 fold 노이즈가 일상적으로 "jump"로 라벨링되고
# 그 노이즈가 검색 부스팅(reflection_impact)에 그대로 들어갔다. 2.0σ로 상향해 방어적
# 기본값으로 삼는다. 대회 데이터가 쌓이면 fold_std 실측 분포로 재캘리브레이션 (ADR-012).
LABEL_Z: float = 2.0

# 승격 cross-seed 확인: 이 seed 목록 전부에서 gain_vs_best > 0 재현돼야 승격
PROMOTE_CONFIRM_SEEDS: list[int] = [
    int(s) for s in os.getenv("PROMOTE_CONFIRM_SEEDS", "7,42,101,137").split(",")
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
