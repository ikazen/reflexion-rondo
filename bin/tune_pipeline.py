"""확정 pipeline(raw.pipelines)의 model_spec/ensemble_spec 멤버를 Optuna로 튜닝한다.

900s attempt CPU 예산 밖 별도 Airflow DAG(reflexion_rondo_tune) 전용 진입점 — 로컬
실행도 가능(장시간 백테스트용). 결과는 raw.tuned_params에 기록되고, 다음 attempt부터
cycle/run.py:_latest_tuned_params가 ctx.tuned_params advisory로 흘려보낸다.

Usage (container/로컬):
    uv run python -m bin.tune_pipeline --competition s4e10 --n-trials 100 --timeout-sec 3600
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _load_confirmed_pipeline_source(conn, competition_id: str) -> tuple[str, float]:
    """확정(cross-seed 통과) pipeline의 materialized 소스를 반환한다 — bin/submit.py의
    자동 선택 경로(_load_best_code)와 동일한 신뢰 기준(raw.pipelines, invalid_reason
    없음). --attempt-id 같은 미확정 escape hatch는 튜닝 대상에서 의도적으로 제외 —
    아직 검증 안 된 코드에 장시간 컴퓨트를 쓰는 건 낭비."""
    row = conn.execute(
        """
        select p.code, p.cv_score
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        where p.competition_id = %s
          and p.cv_score is not null
          and p.invalid_reason is null
        order by c.metric_sign * p.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    if not row:
        raise ValueError(f"No confirmed pipeline for {competition_id} — nothing to tune")
    return row[0].strip(), row[1]


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout-sec", type=int, default=None, help="모델/멤버 1개당 wall-clock 상한")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    import importlib

    from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext
    from evaluator.tuner import tune_confirmed_pipeline
    from store.db import connect, insert_tuned_params
    from store.train_data import load_train

    try:
        comp = importlib.import_module(f"config.competitions.{args.competition}")
    except ModuleNotFoundError as exc:
        print(f"[tune_pipeline] competition config not found: {exc}", file=sys.stderr)
        sys.exit(1)

    conn = connect(apply_schema=False)

    try:
        source, confirmed_cv = _load_confirmed_pipeline_source(conn, comp.COMPETITION_ID)
    except ValueError as exc:
        print(f"[tune_pipeline] {exc}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    ns: dict = {}
    exec(compile(source, "<confirmed_pipeline>", "exec"), ns)  # noqa: S102
    patch_cls = ns.get("Patch")
    if patch_cls is None:
        print("[tune_pipeline] confirmed pipeline source has no Patch class", file=sys.stderr)
        conn.close()
        sys.exit(1)
    pipeline = PatchedPipeline(BasePipeline(), patch_cls())

    train = load_train(comp)
    ctx = PipelineContext(
        target_col=comp.TARGET,
        metric=comp.METRIC,
        n_splits=getattr(comp, "N_SPLITS", 5),
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
    )

    print(f"[tune_pipeline] {comp.COMPETITION_ID} confirmed_cv={confirmed_cv} n_trials={args.n_trials}")

    try:
        results = tune_confirmed_pipeline(
            pipeline, train, ctx, n_trials=args.n_trials, timeout_sec=args.timeout_sec,
        )
    except ValueError as exc:
        print(f"[tune_pipeline] {exc}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    tuning_run_id = str(uuid.uuid4())
    for r in results:
        insert_tuned_params(
            conn,
            id_=str(uuid.uuid4()),
            tuning_run_id=tuning_run_id,
            competition_id=comp.COMPETITION_ID,
            model_type=r.model_name,
            member_index=r.member_index,
            params=r.best_params,
            cv_score=r.best_cv_score,
            baseline_cv_score=r.baseline_cv_score,
            n_trials=r.n_trials,
            improved=r.improved,
        )
        member_tag = f"member[{r.member_index}]" if r.member_index is not None else "single"
        mark = "IMPROVED" if r.improved else "no gain"
        print(
            f"  {member_tag} model={r.model_name} baseline={r.baseline_cv_score:.6f}"
            f" best={r.best_cv_score:.6f} n_trials={r.n_trials} [{mark}]"
        )
    conn.close()


if __name__ == "__main__":
    main()
