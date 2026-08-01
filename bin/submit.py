"""best attempt 코드로 전체 train 학습 → test 예측 → Kaggle 제출.

Usage:
    uv run python -m bin.submit --competition s4e1            # CSV만 생성
    uv run python -m bin.submit --competition s4e1 --submit   # 생성 후 Kaggle 제출
    uv run python -m bin.submit --competition s4e1 --attempt-id <id>  # 특정 attempt 지정
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests

ROOT = Path(__file__).parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")
RUNS_DIR = ROOT / "runs"
CODE_SEP = "# " + "-" * 60

# 5-seed 예측 평균(bagging) — 단일 seed=42 fit 대비 거의 공짜인 LB 이득.
_BAG_SEEDS = [42, 101, 7, 13, 29]

# LightGBM/XGBoost/CatBoost가 생성자에서 받는 early-stopping 관련 키.
# param_candidates()가 CV 경로(harness._fit_with_early_stopping, eval_set 있음)를
# 염두에 두고 넣은 값인데, 이 모듈은 전체 train으로 최종 fit해 eval_set이 없다 —
# LightGBM은 이 키가 있으면 eval_set 유무와 무관하게 조기종료 콜백을 등록해버려
# fit()이 죽는다(#71). _fit_full_train()의 재시도 폴백에서만 제거해 쓴다.
_EARLY_STOPPING_KEYS = frozenset({
    "early_stopping_rounds", "early_stopping_round", "early_stopping",
    "n_iter_no_change", "callbacks", "od_wait", "od_type", "validation_fraction",
})


def _load_best_code(
    competition_id: str, attempt_id: str | None
) -> tuple[str, float, str, str | None, object | None]:
    """(code_source, cv_score, attempt_id, pipeline_sha256, run_ts) 반환.

    attempt_id 지정 시: 사용자가 명시적으로 고른 attempt(미확정이어도 허용) — 의도된 escape
    hatch. pipeline_sha256은 None(무결성 검증 스킵 — raw.pipelines가 아니라 raw.attempts의
    개별 code_path라 대조할 신뢰 해시가 없음, 명시적 지정이므로 허용). run_ts는 이 attempt의
    cv_score가 측정된 시점 — 그 시점까지 승격된 base pipeline을 재생(#80)할 기준점으로 쓴다.
    미지정(자동 선택) 시: confirmed 파이프라인(raw.pipelines, cross-seed 통과분)만
    소스로 쓴다. _load_pipeline()이 실제 제출하는 모델도 confirmed 소스에서 materialize된
    것이므로, 리포팅되는 cv_score/attempt_id도 같은 소스여야 한다. raw.attempts all-time
    max는 미확정 attempt를 가리킬 수 있어 리포팅 불일치를 낳았다. run_ts는 이 경로에선
    불필요(None) — MinIO best_pipeline.py가 이미 그 시점의 base를 담고 있다.
    pipeline_sha256은 MinIO best_pipeline.py 무결성 검증용 신뢰 해시(raw.pipelines,
    materialize 시점 기록).
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    conn = connect(apply_schema=False)

    if attempt_id:
        row = conn.execute(
            "select code_path, cv_score, attempt_id, run_ts from raw.attempts"
            " where attempt_id like %s and competition_id = %s",
            [f"{attempt_id}%", competition_id],
        ).fetchone()
        conn.close()
        if not row:
            raise ValueError(f"No valid attempt found for {competition_id}")
        code_path, cv_score, aid, run_ts = row
        from store.s3_code import download as _code_download
        content = _code_download(code_path)
        if not content:
            raise FileNotFoundError(f"code not found: {code_path}")
        sep = CODE_SEP + "\n"
        source = content.split(sep, 1)[1].strip() if sep in content else content.strip()
        return source, cv_score, aid, None, run_ts

    row = conn.execute(
        """
        select p.code, p.cv_score, p.attempt_id, p.pipeline_sha256
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        where p.competition_id = %s
          and p.cv_score is not null
        order by c.metric_sign * p.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError(
            f"No confirmed pipeline for {competition_id} — "
            "use --attempt-id to submit an unconfirmed attempt explicitly"
        )
    source, cv_score, aid, pipeline_sha256 = row
    return source.strip(), cv_score, aid, pipeline_sha256, None


def _read_csv(comp: object, name: str) -> pl.DataFrame:
    s3 = getattr(comp, "S3_DATA_PATH", None)
    if s3 and _MINIO_ENDPOINT:
        try:
            url = f"{_MINIO_ENDPOINT}/kaggle/{s3}{name}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return pl.read_csv(resp.content)
        except Exception as exc:
            raise FileNotFoundError(
                f"MinIO read failed for {name} ({url}): {exc}"
            ) from exc
    return pl.read_csv(getattr(comp, "DATA_DIR") / name)


_HOOK_NAMES = (
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
)


def _load_pipeline(
    competition_id: str,
    extra_source: str | None = None,
    expected_sha256: str | None = None,
    attempt_only: bool = False,
    base_source: str | None = None,
) -> object:
    """Load the materialized best pipeline from MinIO.

    extra_source: attempt source exec'd first so helper classes defined there
    (e.g. WeightedEnsemble) are available when the stored best pipeline runs.
    Needed for pipelines materialized before ClassDef support existed.

    expected_sha256: MinIO kaggle 버킷은 익명 write 허용이라 best_pipeline.py가
    변조될 수 있다. raw.pipelines(신뢰된 Postgres 사본)에 기록된 해시와 대조해 불일치 시
    조용히 진행하지 않고 즉시 raise한다. None이면 검증 스킵(예: --attempt-id 명시 경로).

    attempt_only: True면 MinIO best_pipeline.py를 아예 조회하지 않고
    extra_source만 단독으로 exec한다. --attempt-id 경로(및 confirmed pipeline 부재 시
    auto-submit 폴백)는 애초에 신뢰 해시가 없어(expected_sha256=None) 위 무결성 검증이
    스킵되는데, raw.pipelines 행이 삭제돼도 대응 MinIO blob은 안 지워지므로 고아
    (orphaned) blob이 같은 이름의 class Patch로 조용히 덮어써 특정 attempt 제출을
    하이재킹할 수 있다. attempt_only=True는 그 무관한 MinIO 상태를 원천적으로 배제한다.

    base_source: attempt_only=True일 때 patch가 그 위에서 실행돼야 할 base pipeline
    소스(cycle.materialize.load_base_snapshot — raw.pipelines.materialized_code
    스냅샷 우선, MinIO가 아니라 Postgres 신뢰 사본이라 위 하이재킹 우려와 무관하다,
    #89). 평가 경로
    (runtime/runner.py:_load_best_pipeline_class)가 base+patch 위에서 cv_score를
    측정하는데, 여기서 base 없이 patch만 실행하면(#80) hook을 하나만 오버라이드하는
    attempt(예: param_candidates만 바꾼 하이퍼파라미터 탐색)가 나머지 hook에서
    BasePipeline 기본 모델로 조용히 떨어져 평가와 전혀 다른(대개 훨씬 나쁜) 예측을
    제출하게 된다. None이면(재생 이력 없음 등) 기존대로 BasePipeline()에 patch만 적용.

    attempt_only 경로는 예전에 훅 메서드만 `type(...)`으로 새 클래스에 옮겨
    붙였는데(methods={h: getattr(patch_cls, h) ...}), Patch가 훅 밖 클래스 속성(예:
    s6e7 우승 attempt의 `_ordinal_orders`)에 의존하면 그 상태가 통째로 소실돼
    AttributeError로 죽었다. 평가 경로(runtime/runner.py)는 실제 Patch() 인스턴스를
    PatchedPipeline으로 감싸 이 문제가 없었으므로 — cv 산출은 통과하고 submit만
    크래시하는 불일치가 생겼다. 여기도 동일하게 PatchedPipeline을 써서 인스턴스 상태를
    보존한다.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from store.s3_code import download_best_pipeline
    from evaluator.harness import BasePipeline, PatchedPipeline

    if attempt_only:
        if not extra_source:
            return BasePipeline()
        ns: dict = {}
        exec(compile(extra_source, "<attempt_pipeline>", "exec"), ns)  # noqa: S102
        patch_cls = ns.get("Patch")
        if not patch_cls:
            return BasePipeline()
        base = BasePipeline()
        if base_source:
            base_ns: dict = {}
            exec(compile(base_source, "<replayed_best_pipeline>", "exec"), base_ns)  # noqa: S102
            base_patch_cls = base_ns.get("Patch")
            if base_patch_cls:
                methods = {h: getattr(base_patch_cls, h) for h in _HOOK_NAMES if hasattr(base_patch_cls, h)}
                base = type("ReplayedBase", (BasePipeline,), methods)()
        return PatchedPipeline(base, patch_cls())

    best_source = download_best_pipeline(competition_id)
    if not best_source:
        # confirmed pipeline(best_pipeline.py) 부재 시(예: s6e6는 merge-verify 크래시로
        # 승격이 막힘) 기존엔 BasePipeline 기본 모델로 폴백해 약한 제출을
        # 냈다. auto_submit이 넘긴 best attempt source(cross-seed confirm 통과분)에
        # Patch가 있으면 그걸로 제출한다 — 이미 격리 평가를 거친 코드라
        # --attempt-id 명시 제출과 동일한 신뢰 수준.
        if not extra_source:
            return BasePipeline()
        best_source = extra_source
        extra_source = None
        expected_sha256 = None  # attempt source엔 신뢰 해시 없음 — 검증 스킵

    if expected_sha256:
        import hashlib
        actual_sha256 = hashlib.sha256(best_source.encode()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"best_pipeline.py integrity check failed for {competition_id}: "
                f"sha256 mismatch (expected {expected_sha256[:12]}…, got {actual_sha256[:12]}…) "
                "— MinIO kaggle 버킷 변조 가능성. 확인 없이 exec를 중단한다."
            )

    ns: dict = {}
    if extra_source:
        exec(compile(extra_source, "<attempt_ns>", "exec"), ns)  # noqa: S102
    exec(compile(best_source, "<best_pipeline>", "exec"), ns)  # noqa: S102
    patch_cls = ns.get("Patch")
    if not patch_cls:
        return BasePipeline()
    methods = {h: getattr(patch_cls, h) for h in _HOOK_NAMES if hasattr(patch_cls, h)}
    BestPipelineCls = type("BestPipeline", (BasePipeline,), methods)
    return BestPipelineCls()


def _predict_raw(model, X, metric_class: str):
    """metric_class(evaluator/metrics.get 3번째 값) 기준으로 proba/label
    분기 — comp.IS_CLASSIFICATION 기준이면 accuracy/f1/qwk 같은 label-metric
    classification 대회에서 predict_proba를 잘못 쓰게 된다.
    """
    if metric_class == "binary_proba":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _submission_value_col(sample_columns: list[str], fallback: str) -> str:
    """제출 값 컬럼명은 sample_submission.csv의 실제 2번째 컬럼을 따른다."""
    return sample_columns[1] if len(sample_columns) > 1 else fallback


def _dummy_target_value(train: pl.DataFrame, target_col: str):
    """더미 타깃은 train 실재값이어야 한다 — synthetic placeholder는 Patch의
    exhaustive target 인코딩에서 매핑 밖 값이라 크래시한다."""
    return train[target_col][0]


def _impute_train_test_median(train_np, test_np):
    """NaN 중앙값 대치를 train/test 대칭 적용. medians는 train 기준."""
    col_medians = np.nanmedian(train_np, axis=0)
    train_mask, test_mask = np.isnan(train_np), np.isnan(test_np)
    if train_mask.any():
        train_np = np.where(train_mask, col_medians, train_np)
        print(f"  NaN {train_mask.sum()}개(train) → 훈련 중앙값으로 대체")
    if test_mask.any():
        test_np = np.where(test_mask, col_medians, test_np)
        print(f"  NaN {test_mask.sum()}개(test) → 훈련 중앙값으로 대체")
    return train_np, test_np


def _fit_full_train(pipeline: object, params: dict, ctx: object, X: np.ndarray, y: np.ndarray) -> object:
    """params로 build_model+fit. eval_set 없이 죽으면(#71) early-stopping 키를
    벗긴 params로 한 번 더 시도한다. HistGradientBoosting/CatBoost처럼 eval_set
    없이도 내부 검증 분할로 동작하는 estimator는 첫 시도가 그대로 성공하므로
    건드리지 않는다 — 실패할 때만 개입해 방법론을 조용히 안 바꾼다.

    build_model은 harness의 _build_model_safe로 호출한다(#94) — 평가 경로(#79)와
    동일한 stale/중복 kwarg 스트립. 이게 빠지면 merge-verify를 통과한 병합본이
    제출에서만 크래시하는 평가/제출 비대칭이 생긴다.
    """
    from evaluator.harness import _build_model_safe
    model = _build_model_safe(pipeline, params, ctx)
    try:
        model.fit(X, y)
        return model
    except Exception:
        stripped = {k: v for k, v in params.items() if k not in _EARLY_STOPPING_KEYS}
        if stripped == params:
            raise
        print(f"  [submit] fit failed with early-stopping params, retrying without: {sorted(set(params) - set(stripped))}")
        model = _build_model_safe(pipeline, stripped, ctx)
        model.fit(X, y)
        return model


def _bagged_predict(
    pipeline: object,
    params: dict,
    X_train_np: np.ndarray,
    y_train: np.ndarray,
    X_test_np: np.ndarray,
    ctx: object,
    metric_class: str,
    bag_seeds: list[int] = _BAG_SEEDS,
) -> np.ndarray:
    """seed별 build_model+fit 반복 후 raw 예측 평균(bagging).

    preprocess/feature_transform은 seed 무관이라 호출부에서 1회만 수행되고,
    이 함수는 이미 변환된 X_train_np/X_test_np를 받아 모델 fit만 반복한다.
    """
    from evaluator.harness import PipelineContext
    bag_preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bag_seed in bag_seeds:
            bag_ctx = PipelineContext(
                target_col=ctx.target_col,
                metric=ctx.metric,
                n_splits=ctx.n_splits,
                seed=bag_seed,
                is_classification=ctx.is_classification,
                prev_best=ctx.prev_best,
                action_type=ctx.action_type,
                best_params=ctx.best_params,
            )
            # 여기선 early stopping(harness._fit_with_early_stopping) 미적용 —
            # 전체 train으로 최종 fit하는 자리라 라벨 있는 held-out validation이 없음
            # (test_np는 unlabeled). 억지로 train 일부를 떼면 최종 제출 방법론 자체가
            # 바뀌므로 이번 범위에서 제외 — CV 경로(harness.py)만 적용.
            # 다만 params가 생성자 인자로 early-stopping 키를 담고 있으면(#71,
            # eval_set 없이 fit이 죽음) _fit_full_train이 그 키만 벗겨 재시도한다.
            model = _fit_full_train(pipeline, params, bag_ctx, X_train_np, y_train)
            bag_preds.append(_predict_raw(model, X_test_np, metric_class))
    if metric_class == "classification":
        # discrete label 예측(멀티클래스 문자열 라벨 포함, s6e6)은 평균이 의미 없다 —
        # 문자열이면 np.mean이 TypeError로 죽고, 정수 인코딩이어도 평균은 라벨이 아니다.
        # seed별 다수결로 집계한다(scipy.stats.mode는 1.11부터 non-numeric dtype 미지원
        # 이라 순수 numpy로 구현). binary_proba/regression_error는 연속값이라 기존
        # 평균(np.mean) 그대로 둔다.
        return _majority_vote(bag_preds)
    return np.mean(bag_preds, axis=0)


def _majority_vote(bag_preds: list[np.ndarray]) -> np.ndarray:
    """seed별 예측(라벨, 문자열 가능) 중 샘플별 최빈값. dtype 무관하게 동작."""
    stacked = np.stack(bag_preds, axis=0)  # (n_seeds, n_samples)
    classes, codes = np.unique(stacked, return_inverse=True)
    codes = codes.reshape(stacked.shape)
    counts = np.stack([(codes == k).sum(axis=0) for k in range(len(classes))], axis=1)
    return classes[counts.argmax(axis=1)]


def generate_submission_csv(
    competition_slug: str, attempt_id: str | None = None
) -> tuple[Path, str, float]:
    """best 코드 로드 → 전체 train 5-seed fit → test 예측 → CSV 저장.

    (csv_path, attempt_id, cv_score) 반환. CLI(main)와 promote 훅(캐시 생성)
    양쪽에서 재사용 — 로직은 하나만 유지.
    """
    import sys
    sys.path.insert(0, str(ROOT))

    comp = importlib.import_module(f"config.competitions.{competition_slug}")

    source, cv_score, resolved_attempt_id, pipeline_sha256, run_ts = _load_best_code(
        comp.COMPETITION_ID, attempt_id
    )
    print(f"best attempt: {resolved_attempt_id[:8]}  cv={cv_score:.5f}")

    # attempt_only 경로(--attempt-id 명시 또는 auto-submit 폴백)는 patch가 그 attempt
    # 평가 시점까지 승격된 base 위에서 실행돼야 cv_score와 같은 예측이 나온다(#80).
    # base는 raw.pipelines.materialized_code(승격 당시 병합본의 Postgres 신뢰 사본)
    # 우선, 스냅샷 없는 과거 이력만 strict replay 폴백(#89) — MinIO 조회는 여전히
    # 하지 않는다(고아 blob 하이재킹 방지, #19/#21 유지).
    base_source = None
    if attempt_id:
        from cycle.materialize import load_base_snapshot
        from store.db import connect as _connect
        _conn = _connect(apply_schema=False)
        try:
            base_source, base_origin = load_base_snapshot(
                _conn, comp.COMPETITION_ID, before_run_ts=run_ts
            )
        finally:
            _conn.close()
        if base_source:
            print(f"base pipeline loaded: {base_origin}")
        else:
            print("no prior promoted pipeline — base is BasePipeline()")

    from evaluator.harness import PipelineContext, preselect_params
    from store.train_data import load_train
    pipeline = _load_pipeline(
        comp.COMPETITION_ID, extra_source=source, expected_sha256=pipeline_sha256,
        attempt_only=bool(attempt_id), base_source=base_source,
    )
    ctx = PipelineContext(
        target_col=comp.TARGET,
        metric=comp.METRIC,
        n_splits=5,
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
    )

    train = load_train(comp)
    test  = _read_csv(comp, "test.csv")

    sample = _read_csv(comp, "sample_submission.csv")
    id_col = sample.columns[0]
    test_ids = test[id_col]
    test_feat = test.drop([c for c in comp.DROP_COLS if c in test.columns])

    test_with_dummy = test_feat.with_columns(
        pl.lit(_dummy_target_value(train, comp.TARGET)).cast(train[comp.TARGET].dtype).alias(comp.TARGET)
    )

    params = preselect_params(pipeline, train, ctx)
    train_proc, test_proc = pipeline.preprocess(train, test_with_dummy, comp.TARGET, ctx)
    X_train, X_test = pipeline.feature_transform(train_proc, test_proc, comp.TARGET, ctx)
    y_train = train_proc[comp.TARGET].to_numpy()

    # CV 경로(harness.py:328-330)와 동일하게: target 제거 후 남은 categorical 인코딩.
    # submit이 이 단계를 빠뜨려 BasePipeline 폴백 시(confirmed pipeline 없음) raw string이
    # astype(float)에서 크래시했다 (s6e6 'M', s5e5 'male').
    from evaluator.harness import _strip_target, _encode_residual_categoricals
    X_train = _strip_target(X_train, comp.TARGET)
    X_test = _strip_target(X_test, comp.TARGET)
    X_train, X_test = _encode_residual_categoricals(X_train, X_test)

    X_train_np, X_test_np = _impute_train_test_median(
        X_train.to_numpy().astype(float), X_test.to_numpy().astype(float)
    )

    from evaluator.metrics import get as get_metric
    _, _, metric_class = get_metric(comp.METRIC)

    raw_preds = _bagged_predict(pipeline, params, X_train_np, y_train, X_test_np, ctx, metric_class)
    preds = pipeline.postprocess_predictions(raw_preds, ctx)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"submission_{competition_slug}_{ts}.csv"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    value_col = _submission_value_col(sample.columns, comp.TARGET)
    pl.DataFrame({id_col: test_ids, value_col: preds}).write_csv(out)
    print(f"submission saved: {out}")

    return out, resolved_attempt_id, cv_score


def upload_csv_to_kaggle(competition_id: str, csv_path: Path, message: str) -> subprocess.CompletedProcess:
    """생성된 CSV를 Kaggle에 제출. subprocess 결과(returncode/stdout/stderr) 그대로 반환."""
    return subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submit",
         "-c", competition_id, "-f", str(csv_path), "-m", message],
        capture_output=True, text=True,
    )


def main() -> None:
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--attempt-id", default=None, help="특정 attempt ID 앞 8자리 (참고용)")
    parser.add_argument("--submit", action="store_true", help="Kaggle에 바로 제출")
    parser.add_argument("--message", "-m", default=None, help="제출 메시지")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    comp = importlib.import_module(f"config.competitions.{args.competition}")

    out, attempt_id, cv_score = generate_submission_csv(args.competition, args.attempt_id)

    if args.submit:
        msg = args.message or f"reflexion best cv={cv_score:.5f} attempt={attempt_id[:8]}"
        result = upload_csv_to_kaggle(comp.COMPETITION_ID, out, msg)
        print(result.stdout or result.stderr)
        if result.returncode != 0:
            print(f"kaggle submit failed (rc={result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        print(f"\n제출하려면: uv run python -m bin.submit --competition {args.competition} --submit")


if __name__ == "__main__":
    main()
