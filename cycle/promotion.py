"""승격 후보 cross-seed 확인 + audit holdout 측정.

1σ 게이트(is_significant_gain) 통과 후 이 모듈을 호출한다.
confirm_and_measure가 confirmed=True를 반환해야 insert_pipeline/materialize 진행.
승격 경로 4곳(cycle/run.py x2, bin/run_promote_task.py, bin/establish_baseline.py)에서
공유한다.

PromotionCache(선택)가 주어지면 (1) 동일 행동 지문의 확정 거부 판정을 재계산하지
않고(negative-only memo) (2) 변하지 않은 best pipeline baseline eval을 캐시한다 —
둘 다 안 주면(기본값) 기존 동작과 동일. #166/#167/#168 배경은 docs/decisions.md 참고.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

import polars as pl

from evaluator.metrics import get as get_metric
from runtime.isolate import eval_isolated

_LOG = logging.getLogger(__name__)

# best_source가 없을 때 베이스라인 평가에 쓰는 기본 패치 (= BasePipeline 그대로)
_NOOP_PATCH = "class Patch:\n    pass\n"


@dataclass(frozen=True, slots=True)
class ConfirmResult:
    confirmed: bool
    holdout_score: float | None
    seed_gains: dict | None = field(default=None)
    # 현재 best(또는 콜드스타트면 BasePipeline) 대비 holdout이 악화됐는지.
    # confirmed는 이미 이 값을 반영해 AND 결합돼 있다 — 별도 필드로 노출하는 건
    # 승격 거부 사유(cross-seed 미재현 vs holdout 악화)를 로그/DB에서 구분하기 위함.
    holdout_regressed: bool = False


# --- confirm 게이트 캐시 (raw.confirm_memo / raw.baseline_eval_cache) ---
#
# 2026-08 실측(s6e1): confirm 39회가 (cv_score, fold_scores) 행동 지문 기준
# 4그룹으로 붕괴 — 35회가 이미 확정된 거부 판정을 재계산했다. 소스 기반 dedupe는
# 안 통한다(같은 cv_score를 내는 34개 후보가 서로 다른 AST) — 반드시 행동
# 지문이어야 한다. 별도로, 8회 eval 중 4회는 변하지 않는 best pipeline
# baseline이라 후보와 무관하게 캐시 가능.
_CACHE_TTL = "30 days"


def _train_fingerprint(train90: pl.DataFrame) -> str:
    """train90의 컬럼명+dtype 조합. comp.MAX_TRAIN_ROWS/DROP_COLS/EXTRA_TRAIN_PATHS
    등 대회 데이터 설정이 바뀌면 여기(폭·타입) 또는 height가 달라져 캐시가
    자동 무효화된다 — comp 모듈 접근 없이 train90 자체만으로 판별 가능."""
    return "|".join(f"{c}:{train90.schema[c]}" for c in sorted(train90.columns))


def _hash_key(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _eval_context_key(
    *,
    competition_id: str,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
) -> tuple:
    """confirm_and_measure 호출을 식별하는 공통 키 요소 — baseline 캐시와 confirm
    memo가 공유한다. best_source가 바뀌면(=승격 발생) 자동으로 다른 키가 된다."""
    return (
        competition_id,
        hashlib.sha256((best_source or "").encode()).hexdigest(),
        train90.height,
        _train_fingerprint(train90),
        target_col,
        metric,
        n_splits,
        is_classification,
    )


def _rounded_signature(cv: float, fold_scores: list[float] | None) -> tuple:
    """ULP 흔들림을 흡수하는 행동 지문. 9자리 반올림은 관측된 서로 다른 후보 간
    최소 차이(5.4e-6)보다 훨씬 촘촘해 과병합 위험이 없다(조사 단계 실측)."""
    return (round(cv, 9), tuple(round(f, 9) for f in (fold_scores or [])))


def _rejected_by_error(seed_gains: dict) -> bool:
    """cross-seed 거부가 (baseline 또는 candidate) eval 에러 때문이었는지 판별.
    측정값 기반 거부(gain<=0)와 구분해야 한다 — 에러 기반 거부는 memo에 담지
    않고 holdout 측정도 스킵한다(일시적 크래시를 영구 판정으로 굳히지 않기 위해,
    ADR-027/028)."""
    return any(v.get("error") for v in seed_gains.values())


class PromotionCache:
    """confirm 게이트 캐시 — conn을 감싸는 얇은 래퍼.

    두 종류를 관리한다:
    - baseline eval 캐시(raw.baseline_eval_cache): 변하지 않은 best pipeline을
      후보마다 재평가하지 않는다. 성공한 평가만 저장 — 에러는 캐시하지 않는다.
    - confirm memo(raw.confirm_memo): 이미 확정된 거부 판정을 재계산하지 않는다.
      negative-only(승격은 캐시 안 함) + 에러 기반 거부도 저장 안 함.

    둘 다 raw.attempts/raw.pipelines가 아닌 전용 테이블 — 승격 판정의 중간
    산출물이지 관측 대상 자체가 아니다. TTL 경과분은 무시하고 재계산한다.
    """

    __slots__ = ("conn",)

    def __init__(self, conn) -> None:
        self.conn = conn

    def get_baseline(self, key: tuple, mode: str, seed: int) -> float | None:
        row = self.conn.execute(
            f"""select score from raw.baseline_eval_cache
                where cache_key = %s and created_at > now() - interval '{_CACHE_TTL}'""",
            [_hash_key(*key, mode, seed)],
        ).fetchone()
        return row[0] if row else None

    def put_baseline(self, key: tuple, mode: str, seed: int, competition_id: str, score: float) -> None:
        self.conn.execute(
            """
            insert into raw.baseline_eval_cache (cache_key, competition_id, mode, seed, score)
            values (%s, %s, %s, %s, %s)
            on conflict (cache_key) do update set score = excluded.score, created_at = now()
            """,
            [_hash_key(*key, mode, seed), competition_id, mode, seed, score],
        )

    def get_memo(
        self,
        key: tuple,
        candidate_cv: float,
        candidate_fold_scores: list[float] | None,
        confirm_seeds: list[int],
    ) -> ConfirmResult | None:
        memo_key = _hash_key(*key, tuple(confirm_seeds), *_rounded_signature(candidate_cv, candidate_fold_scores))
        row = self.conn.execute(
            f"""select holdout_score, holdout_regressed, seed_gains from raw.confirm_memo
                where memo_key = %s and created_at > now() - interval '{_CACHE_TTL}'""",
            [memo_key],
        ).fetchone()
        if not row:
            return None
        holdout_score, holdout_regressed, seed_gains = row
        return ConfirmResult(
            confirmed=False,
            holdout_score=holdout_score,
            seed_gains=seed_gains,
            holdout_regressed=bool(holdout_regressed),
        )

    def put_memo(
        self,
        key: tuple,
        candidate_cv: float,
        candidate_fold_scores: list[float] | None,
        confirm_seeds: list[int],
        competition_id: str,
        result: ConfirmResult,
    ) -> None:
        memo_key = _hash_key(*key, tuple(confirm_seeds), *_rounded_signature(candidate_cv, candidate_fold_scores))
        self.conn.execute(
            """
            insert into raw.confirm_memo
                (memo_key, competition_id, cv_score, fold_scores, holdout_score, holdout_regressed, seed_gains)
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (memo_key) do update set
                holdout_score = excluded.holdout_score,
                holdout_regressed = excluded.holdout_regressed,
                seed_gains = excluded.seed_gains,
                created_at = now()
            """,
            [
                memo_key, competition_id, candidate_cv,
                json.dumps(candidate_fold_scores) if candidate_fold_scores is not None else None,
                result.holdout_score, result.holdout_regressed,
                json.dumps(result.seed_gains) if result.seed_gains else None,
            ],
        )


def effective_label(original_label: str, confirm: ConfirmResult | None) -> str:
    """confirm이 jump를 거부하면(cross-seed 미재현 또는 holdout 악화) bandit
    보상·reflection lesson에는 regression으로 반영한다 — CV 단계에서만 좋아
    보였을 뿐 실제 검증(cross-seed 재현/holdout)은 통과 못 한 방향이라는 뜻이므로.

    이 구분이 없으면 update_bandit이 attempt 생성 시점의 잠정 label(confirm
    이전)만 보고 α+=1.0을 준다 — confirm이 나중에 거부해도 그 보상은 되돌아가지
    않아, 같은 아이디어가 계속 높은 확률로 재선택되는 자기강화 루프가 생긴다
    (#164 실측: s6e1의 preprocessing 후보가 cv_score 소수점 10자리까지 동일하게
    32회 재생성 — 매번 holdout에서 거부됐지만 bandit은 그때마다 최댓값 보상을 받음).

    confirm=None(스킵됨)이거나 원본이 이미 jump가 아니면 그대로 반환 — jump만
    다운그레이드 대상이다(neutral/regression/error는 confirm을 애초에 안 탄다).
    """
    if original_label == "jump" and confirm is not None and not confirm.confirmed:
        return "regression"
    return original_label


def confirm_and_measure(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    holdout10: pl.DataFrame | None,
    target_col: str,
    metric: str,
    n_splits: int,
    seed: int,
    is_classification: bool,
    confirm_seeds: list[int],
    action_type: str = "",
    cache: PromotionCache | None = None,
    competition_id: str | None = None,
    candidate_cv: float | None = None,
    candidate_fold_scores: list[float] | None = None,
) -> ConfirmResult:
    """Cross-seed paired 재현 확인 + audit holdout 1회 측정·게이트.

    cross-seed: 각 seed에서 베이스라인(best pipeline)도 같은 seed로 재평가해
    paired gain(candidate@seed - baseline@seed) > 0이어야 confirmed=True. seed만
    바꾼 CV라 preprocess의 valid-target 의존 누수처럼 seed 불변인 문제는 못 잡는다.
    holdout: holdout10 있으면 train90으로 fit, holdout10 측정 → holdout_score.
    후보뿐 아니라 현재 best(콜드스타트면 BasePipeline)도 같은 holdout으로 측정해
    비교한다 — 후보가 더 나쁘면(holdout_regressed) confirmed를 강제로 False로
    떨어뜨린다. _eval_holdout(runtime/runner.py)이 dummy target으로 실제 추론
    조건을 재현하므로 cross-seed가 못 잡는 누수를 값 자체로 걸러낼 수 있다.
    baseline holdout을 측정 못 하면(에러) 비교 근거가 없으므로 보수적으로
    regressed=False로 두고 막지 않는다 — 정보 없음과 악화 확인은 다르다.

    cache/competition_id/candidate_cv가 모두 주어지면 두 단계로 재계산을 줄인다:
    (1) 동일 행동 지문(candidate_cv+candidate_fold_scores)의 확정 거부 판정이
    이미 memo에 있으면 eval 없이 그 결과를 재사용한다. (2) best pipeline
    baseline eval을 캐시해 후보마다 재계산하지 않는다. 인자를 안 주면(기본값
    전부 None) 기존 동작과 완전히 동일하다.
    """
    ctx_key: tuple | None = None
    if cache is not None and competition_id is not None:
        ctx_key = _eval_context_key(
            competition_id=competition_id, best_source=best_source, train90=train90,
            target_col=target_col, metric=metric, n_splits=n_splits,
            is_classification=is_classification,
        )
        if candidate_cv is not None:
            memo = cache.get_memo(ctx_key, candidate_cv, candidate_fold_scores, confirm_seeds)
            if memo is not None:
                _LOG.info(
                    "confirm memo hit — 동일 행동 지문의 확정 거부 판정 재사용 (cv=%.6f, eval 0회)",
                    candidate_cv,
                )
                return memo

    confirmed, seed_gains = _cross_seed_confirm(
        source=source,
        best_source=best_source,
        train90=train90,
        target_col=target_col,
        metric=metric,
        n_splits=n_splits,
        is_classification=is_classification,
        confirm_seeds=confirm_seeds,
        action_type=action_type,
        cache=cache,
        ctx_key=ctx_key,
        competition_id=competition_id,
    )

    # cross-seed가 candidate eval 에러로 거부했으면 holdout도 같은 후보를 평가하는
    # 것이라 재현 안 될 확률이 높고, 설령 결과가 나와도 confirmed는 이미 False로
    # 확정돼 있어 판정을 못 바꾼다 — 실측(s6e8): cross-seed candidate 크래시 후에도
    # holdout candidate eval에 8.3분을 더 쓰고 결국 에러로 종료. 측정값 기반 거부
    # (gain<=0)는 holdout_score가 overfit_gap 드리프트 관측(holdout_cv_gap_trend
    # 뷰 등 여러 소비처)에 여전히 쓰이므로 그대로 측정한다.
    cross_seed_errored = _rejected_by_error(seed_gains)

    holdout_score: float | None = None
    holdout_regressed = False
    if holdout10 is not None and not cross_seed_errored:
        holdout_score = _measure_holdout(
            source=source,
            best_source=best_source,
            train90=train90,
            holdout10=holdout10,
            target_col=target_col,
            metric=metric,
            n_splits=n_splits,
            seed=seed,
            is_classification=is_classification,
            action_type=action_type,
        )
        if holdout_score is not None:
            baseline_holdout_score = cache.get_baseline(ctx_key, "holdout", seed) if ctx_key is not None else None
            if baseline_holdout_score is None:
                baseline_holdout_score = _measure_holdout(
                    source=best_source if best_source else _NOOP_PATCH,
                    best_source=None,
                    train90=train90,
                    holdout10=holdout10,
                    target_col=target_col,
                    metric=metric,
                    n_splits=n_splits,
                    seed=seed,
                    is_classification=is_classification,
                    action_type=action_type,
                )
                if baseline_holdout_score is not None and ctx_key is not None:
                    cache.put_baseline(ctx_key, "holdout", seed, competition_id, baseline_holdout_score)
            if baseline_holdout_score is not None:
                _, metric_sign, _ = get_metric(metric)
                holdout_regressed = (
                    metric_sign * holdout_score < metric_sign * baseline_holdout_score
                )
                if holdout_regressed:
                    _LOG.warning(
                        "holdout 악화로 승격 거부: candidate=%.6f baseline=%.6f",
                        holdout_score, baseline_holdout_score,
                    )

    result = ConfirmResult(
        confirmed=confirmed and not holdout_regressed,
        holdout_score=holdout_score,
        seed_gains=seed_gains if seed_gains else None,
        holdout_regressed=holdout_regressed,
    )

    # negative-only: 확정 승격은 캐시하지 않고(같은 코드가 재현될 일이 없다),
    # 에러 기반 거부도 캐시하지 않는다(cross_seed_errored 정의부 주석 참고).
    if (
        ctx_key is not None and candidate_cv is not None
        and not result.confirmed and not cross_seed_errored
    ):
        cache.put_memo(ctx_key, candidate_cv, candidate_fold_scores, confirm_seeds, competition_id, result)

    return result


_ERROR_TRUNCATE_LEN = 500


def _baseline_cv(
    *,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    seed: int,
    action_type: str,
    cache: PromotionCache | None = None,
    ctx_key: tuple | None = None,
    competition_id: str | None = None,
) -> tuple[float | None, str | None]:
    """best pipeline(또는 BasePipeline)을 seed 고정으로 단독 평가해 (cv_score, error) 반환.

    에러 시 cv_score=None → 호출부에서 보수적으로 승격 거부. error는 seed_gains에
    남겨 confirm 실패가 "재현 안 됨"인지 "크래시"인지 DB만 봐서도 구분 가능하게 한다.

    cache/ctx_key가 있으면 성공한 평가만 캐시하고 재사용한다 — 실패(에러)는
    캐시하지 않는다(일시적 OOM/CPU-kill을 영구 실패로 굳히지 않기 위해).
    """
    if cache is not None and ctx_key is not None:
        cached = cache.get_baseline(ctx_key, "cv", seed)
        if cached is not None:
            return cached, None

    src = best_source if best_source else _NOOP_PATCH
    res = eval_isolated(
        source=src,
        train=train90,
        target_col=target_col,
        metric=metric,
        prev_best=None,
        n_splits=n_splits,
        seed=seed,
        is_classification=is_classification,
        action_type=action_type,
        best_source=None,
    )
    if res.error_trace or res.cv_score is None:
        _LOG.warning("baseline eval failed seed=%d err=%s", seed, bool(res.error_trace))
        err = (res.error_trace or "unknown (cv_score is None with no error_trace)")
        return None, err[:_ERROR_TRUNCATE_LEN]

    if cache is not None and ctx_key is not None and competition_id is not None:
        cache.put_baseline(ctx_key, "cv", seed, competition_id, res.cv_score)
    return res.cv_score, None


def _cross_seed_confirm(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    confirm_seeds: list[int],
    action_type: str,
    cache: PromotionCache | None = None,
    ctx_key: tuple | None = None,
    competition_id: str | None = None,
) -> tuple[bool, dict]:
    if not confirm_seeds:
        return True, {}

    seed_gains: dict = {}

    for cseed in confirm_seeds:
        base_cv, base_err = _baseline_cv(
            best_source=best_source,
            train90=train90,
            target_col=target_col,
            metric=metric,
            n_splits=n_splits,
            is_classification=is_classification,
            seed=cseed,
            action_type=action_type,
            cache=cache,
            ctx_key=ctx_key,
            competition_id=competition_id,
        )
        if base_cv is None:
            _LOG.warning("cross-seed=%d baseline eval 실패 → 승격 취소: %s", cseed, base_err)
            seed_gains[str(cseed)] = {
                "baseline_cv": None,
                "candidate_cv": None,
                "gain": None,
                "error": f"baseline: {base_err}",
            }
            return False, seed_gains

        cand = eval_isolated(
            source=source,
            train=train90,
            target_col=target_col,
            metric=metric,
            prev_best=base_cv,
            n_splits=n_splits,
            seed=cseed,
            is_classification=is_classification,
            action_type=action_type,
            best_source=best_source,
        )

        seed_gains[str(cseed)] = {
            "baseline_cv": base_cv,
            "candidate_cv": cand.cv_score,
            "gain": cand.gain_vs_best,
            "error": (cand.error_trace or "")[:_ERROR_TRUNCATE_LEN] or None,
        }

        if cand.error_trace or cand.gain_vs_best is None or cand.gain_vs_best <= 0:
            _LOG.warning(
                "cross-seed=%d 미재현 → 승격 취소 (baseline=%.6f candidate=%s gain=%s err=%s)",
                cseed, base_cv, cand.cv_score, cand.gain_vs_best, bool(cand.error_trace),
            )
            return False, seed_gains
        _LOG.info(
            "cross-seed=%d 재현 baseline=%.6f candidate=%.6f gain=%+.6f",
            cseed, base_cv, cand.cv_score, cand.gain_vs_best,
        )

    return True, seed_gains


def _measure_holdout(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    holdout10: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    seed: int,
    is_classification: bool,
    action_type: str,
) -> float | None:
    result = eval_isolated(
        source=source,
        train=train90,
        target_col=target_col,
        metric=metric,
        prev_best=None,
        n_splits=n_splits,
        seed=seed,
        is_classification=is_classification,
        action_type=action_type,
        best_source=best_source,
        holdout_data=holdout10,
    )
    if result.holdout_score is not None:
        _LOG.info("holdout_score=%.6f", result.holdout_score)
    return result.holdout_score
