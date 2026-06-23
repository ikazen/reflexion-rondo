"""Super-cycle: 1 shared retrieve + 3 parallel attempts + promote_winner (BON-100).

Promotion rule (D2-a): max gain_vs_best among non-errored attempts.
If all errored: no promoted attempt (winner=None).
Stagnation: only promoted (was_promoted=True / NULL=legacy) attempts count.
Reflect: winner only (BON-96 gate still applies inside _do_reflect).
Bandit: all 3 attempts update (more signal per super-cycle).
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from cycle.run import (
    CycleConfig,
    CycleResult,
    _AttemptData,
    _do_reflect,
    _build_retrieval_query,
    _prev_best,
    _recent_failure_summary,
    run_attempt_core,
)
from cycle.materialize import materialize_best_pipeline
from evaluator.harness import is_significant_gain
from memory.retriever import EmbeddingUnavailableError, search
from store.db import PgConn, connect, insert_pipeline
from store.s3_code import upload_best_pipeline, download_best_pipeline

_N_PARALLEL = 3


@dataclass
class SuperCycleResult:
    super_cycle_id: str
    winner: CycleResult | None
    all_results: list[CycleResult]

    @property
    def attempt_id(self) -> str:
        return self.winner.attempt_id if self.winner else self.super_cycle_id

    @property
    def cv_score(self) -> float | None:
        return self.winner.cv_score if self.winner else None

    @property
    def label(self) -> str:
        return self.winner.label if self.winner else "error"


def _pick_winner(all_data: list[_AttemptData]) -> int | None:
    """Pick the attempt with the best gain_vs_best. Falls back to best cv_score if no gains."""
    with_gain = [(i, d.gain_vs_best) for i, d in enumerate(all_data) if d.gain_vs_best is not None]
    if with_gain:
        return max(with_gain, key=lambda x: x[1])[0]
    return next((i for i, d in enumerate(all_data) if d.cv_score is not None), None)


def run_super_cycle(conn: PgConn, config: CycleConfig) -> SuperCycleResult:
    super_cycle_id = str(uuid.uuid4())

    # 1. Shared retrieve (once)
    fail_summary = _recent_failure_summary(conn, config.competition_id)
    query = _build_retrieval_query(conn, config.competition_id, config.eda_card, fail_summary)
    try:
        lessons = search(conn, query, config.competition_id, k=config.k_retrieve)
    except EmbeddingUnavailableError as exc:
        print(f"[super_cycle] embedding unavailable, proceeding with no lessons: {exc}")
        lessons = []
    prev_best_cv = _prev_best(conn, config.competition_id)

    # 2. 3 parallel attempts — each gets its own DB connection from the pool
    all_data: list[_AttemptData] = []

    def _attempt_task() -> _AttemptData:
        attempt_conn = connect(apply_schema=False)
        try:
            return run_attempt_core(
                attempt_conn, config, lessons, prev_best_cv,
                super_cycle_id=super_cycle_id,
                defer_promotion=True,
            )
        finally:
            attempt_conn.close()

    with ThreadPoolExecutor(max_workers=_N_PARALLEL) as executor:
        futures = [executor.submit(_attempt_task) for _ in range(_N_PARALLEL)]
        for f in as_completed(futures):
            try:
                all_data.append(f.result())
            except Exception as exc:
                print(f"[super_cycle] attempt raised: {exc}")

    if not all_data:
        return SuperCycleResult(super_cycle_id=super_cycle_id, winner=None, all_results=[])

    # 3. Promote winner — update was_promoted for all attempts, then winner-only pipeline record
    winner_idx = _pick_winner(all_data)
    for i, d in enumerate(all_data):
        conn.execute(
            "UPDATE raw.attempts SET was_promoted = %s WHERE attempt_id = %s",
            [i == winner_idx, d.attempt_id],
        )

    if winner_idx is not None:
        wd = all_data[winner_idx]
        if is_significant_gain(wd.gain_vs_best, wd.cv_fold_var) and not wd.error_trace:
            import json
            import uuid as _uuid
            fp_row = conn.execute(
                "select fingerprint from raw.competitions where competition_id = %s",
                [config.competition_id],
            ).fetchone()
            fp_val = fp_row[0] if fp_row and fp_row[0] else {}
            fp_dict = fp_val if isinstance(fp_val, dict) else json.loads(fp_val)
            with conn.transaction():
                insert_pipeline(
                    conn,
                    pipeline_id=str(_uuid.uuid4()),
                    attempt_id=wd.attempt_id,
                    competition_id=config.competition_id,
                    fingerprint_snapshot=fp_dict,
                    code=wd.source,
                    cv_score=wd.cv_score,
                    gain_vs_best=wd.gain_vs_best,
                )
            current_best = download_best_pipeline(config.competition_id)
            materialized = materialize_best_pipeline(current_best, wd.source)
            upload_best_pipeline(config.competition_id, materialized)
            print(f"[super_cycle] winner promoted (gain={wd.gain_vs_best:+.5f})")

    # 4. Build results — reflect winner only
    all_results: list[CycleResult] = []
    winner_result: CycleResult | None = None

    for i, d in enumerate(all_data):
        if i == winner_idx:
            reflection_id = _do_reflect(conn, config.competition_id, d)
            result = CycleResult(
                attempt_id=d.attempt_id,
                cv_score=d.cv_score,
                label=d.label,
                gain_vs_best=d.gain_vs_best,
                retries=d.retries,
                reflection_id=reflection_id,
                error_trace=d.error_trace,
                code_path=d.code_path,
            )
            winner_result = result
        else:
            result = CycleResult(
                attempt_id=d.attempt_id,
                cv_score=d.cv_score,
                label=d.label,
                gain_vs_best=d.gain_vs_best,
                retries=d.retries,
                reflection_id=None,
                error_trace=d.error_trace,
                code_path=d.code_path,
            )
        all_results.append(result)

    return SuperCycleResult(
        super_cycle_id=super_cycle_id,
        winner=winner_result,
        all_results=all_results,
    )
