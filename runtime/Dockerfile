FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "polars>=1.41.2" \
    "scikit-learn>=1.8.0" \
    "lightgbm>=4.6.0" \
    "xgboost>=3.2.0" \
    "catboost>=1.2.10" \
    "numpy" \
    "pandas>=3.0.3" \
    "imbalanced-learn>=0.13.0" \
    "optuna>=4.3.0"

COPY evaluator/ /app/evaluator/
COPY runtime/runner.py /app/runner.py

CMD ["python", "/app/runner.py"]
