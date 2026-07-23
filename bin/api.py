"""reflexion-rondo daemon FastAPI 앱.

엔드포인트:
  GET  /api/heartbeat
  GET  /api/health
  GET  /api/competitions
  POST /api/competitions
  GET  /api/attempts
  GET  /api/attempts/{id}
  GET  /api/lessons
  GET  /api/cold-start
  GET  /api/score/timeline           -- 관측 T3  (#67): CV 추이 + jump 마커 + holdout
  GET  /api/reflexion-health         -- 관측 T5  (#67): 건강 신호등 4칸(#11 §6)
  GET  /api/bandit/posteriors        -- 관측 B2a (#67): posterior + 실측 가중성공률 괴리
  GET  /api/lessons/funnel           -- 관측 B3f (#67): 작성→검색→인용→양의gain 전환율
  GET  /api/lessons/dead             -- 관측 B3d (#67): 인용 0 / 음의gain 교훈
  GET  /api/lessons/duplicates       -- 관측 B3u (#67): near-duplicate 교훈 쌍
  GET  /api/errors/signatures        -- 관측 B4s (#67): 에러 시그니처·재발
  GET  /api/transfer/matrix          -- 관측 X1  (#67): 대회 간 교훈 인용 매트릭스
  GET  /api/competitions/summary     -- 관측 T1  (#67): 대회 목록 + best_cv·상태 롤업
  GET  /api/score                    -- 관측 T2  (#67): 점수 헤드라인(best cv/holdout/gap)
  GET  /api/best-strategy            -- 관측 T4  (#67): best pipeline 구성·인용 교훈
  GET  /api/bandit/timeline          -- 관측 B2b (#67): posterior 리플레이 시계열
  GET  /api/bandit/selection         -- 관측 B2c (#67): 배정 action vs posterior 순위 concordance
  GET  /api/lessons/generality-mix   -- 관측 B3g (#67): L1/L2/L3 비율 시계열
  GET  /api/errors/rate-timeline     -- 관측 B4r (#67): 슈퍼사이클별 에러율 추이
  GET  /api/errors/repeat-offenders  -- 관측 B4o (#67): pitfall 활성 후 재발 시그니처
  GET  /api/promotions               -- 관측 B5  (#67): 승격 타임라인(누적 best)
  GET  /api/transfer/lessons         -- 관측 X2  (#67): 범용 교훈 재사용 리더보드
  GET  /api/transfer/fp-distance     -- 관측 X3  (#67): fingerprint 거리 vs cold-start 이득
  GET  /api/queue
  POST /api/queue
  PATCH /api/queue/{id}
  POST /api/submissions
  POST /api/submissions/auto
  GET  /api/submissions
  GET  /api/submissions/{id}

관측 엔드포인트는 GH #11/#67 설계 24개 전체(대시보드 패널 #65는 후속). 데이터 소스는
store/schema.sql의 파생 뷰(lesson_funnel/lesson_dead/lesson_duplicates/bandit_calibration/
error_recurrence/transfer_matrix) + raw.pipelines/super_cycle_context 직접 조회 + 밴딧
리플레이(_replay_bandit_timeline, 히스토리 테이블 없이 update_bandit 규칙 재현) +
memory.transfer._fp_distance 재사용. retrieved_ids(P1)는 forward-only라 과거 데이터가 섞인
경우 lesson_funnel.retrieved_precise=false로 표시된다.

Postgres 연결은 호출 측(daemon)이 주입한다.
DaemonState는 daemon 메인 루프가 갱신하고 API가 읽는 공유 객체.
"""
from __future__ import annotations

import contextlib
import csv as csv_module
import importlib
import io
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import ACTION_TYPES
from cycle.stagnation import detect_stagnation
from memory.transfer import _fp_distance
from store.db import PgConn

ROOT = Path(__file__).resolve().parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


class _TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> tuple[object, bool]:
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() - entry[0] < ttl:
                return entry[1], True
        return None, False

    def set(self, key: str, val: object) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), val)

    def drop(self, prefix: str) -> None:
        with self._lock:
            for k in [k for k in self._store if k.startswith(prefix)]:
                del self._store[k]


_cache = _TTLCache()


def _traffic_light(*, green: bool, red: bool) -> str:
    """#11 §6 신호등 판정 — red가 green보다 우선."""
    if red:
        return "red"
    if green:
        return "green"
    return "amber"


def _replay_bandit_timeline(conn: PgConn, competition_id: str) -> list[dict]:
    """raw.action_bandit는 현재 alpha/beta만 저장(이력 없음) — reflexion attempts를
    run_ts 순으로 훑으며 cycle/action_optimizer.update_bandit()과 동일한 델타/decay
    규칙을 그대로 재현해 posterior_mean 시계열을 재구성한다(B2b/B2c 공용).

    검증(2026-07 s4e1 기준): 5개 action_type 중 4개는 raw.action_bandit 라이브 값과
    소수점 5자리까지 일치. 1개(feature_engineering)는 유의미하게 어긋났는데 원인은
    raw.attempts에 남지 않은 이력(수동 개입·과거 리셋 등)으로 추정 — attempts 테이블
    바깥의 개입은 이 리플레이가 재현할 수 없다. 추세/방향성 참고용으로 쓰고 절대값을
    라이브 action_bandit_posterior 대체재로 쓰지 말 것."""
    rows = conn.execute(
        """
        select run_ts, action_type, label, gain_vs_best, error_trace
        from raw.attempts
        where competition_id = %s and stage = 'reflexion' and action_type is not null
        order by run_ts
        """,
        [competition_id],
    ).fetchall()
    state: dict[str, tuple[float, float]] = {}
    timeline: list[dict] = []
    for step, (run_ts, action_type, label, gain, err) in enumerate(rows, start=1):
        if action_type not in ACTION_TYPES:
            continue
        # update_bandit(cycle/action_optimizer.py:45-52)과 동일 우선순위 —
        # error/regression이 label='jump'보다 먼저 체크된다.
        if err is not None or label == "regression":
            da, db = 0.0, 1.0
        elif label == "jump":
            da, db = 1.0, 0.0
        elif gain is not None and gain > 0:
            da, db = 0.5, 0.1
        else:
            da, db = 0.1, 0.1
        if action_type not in state:
            alpha, beta = 1.0 + da, 1.0 + db
        else:
            old_a, old_b = state[action_type]
            alpha = 1.0 + (old_a - 1.0) * 0.95 + da
            beta = 1.0 + (old_b - 1.0) * 0.95 + db
        state[action_type] = (alpha, beta)
        timeline.append({
            "step": step,
            "run_ts": run_ts,
            "action_type": action_type,
            "posterior_mean": round(alpha / (alpha + beta), 4),
        })
    return timeline


def _posterior_rank_at(timeline: list[dict], cutoff_ts) -> list[str]:
    """timeline에서 cutoff_ts 이전 각 action_type의 최신 posterior_mean으로 내림차순 순위."""
    latest: dict[str, float] = {}
    for row in timeline:
        if row["run_ts"] <= cutoff_ts:
            latest[row["action_type"]] = row["posterior_mean"]
    return sorted(latest, key=lambda a: latest[a], reverse=True)


@dataclass
class DaemonState:
    current_queue_id: str | None = None
    current_competition: str | None = None
    current_cycle: int = 0
    current_n_cycles: int = 0
    last_cycle_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_queue_id": self.current_queue_id,
                "current_competition": self.current_competition,
                "current_cycle": self.current_cycle,
                "current_n_cycles": self.current_n_cycles,
                "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            }


class RegisterRequest(BaseModel):
    competition: str


class EnqueueRequest(BaseModel):
    competition: str
    stage: str = "reflexion"
    n_cycles: int = 1
    priority: int = 0


class QueuePatchRequest(BaseModel):
    priority: int | None = None
    status: Literal["cancelled"] | None = None


class SubmitRequest(BaseModel):
    competition: str
    attempt_id: str | None = None
    message: str | None = None


class AutoSubmitRequest(BaseModel):
    window_hours: int = 24


_TERMINAL = frozenset({"complete", "error", "invalid"})
# eval 타임아웃은 600->1200s로 올렸으나 submit은 누락 — s5e5(75만 행) 5-seed
# bagging이 600s를 넘겨 매번 타임아웃으로 실패했다. eval과 동일하게 상향.
_SUBMIT_TIMEOUT_SEC = 1200


def _run_in_pgroup(
    cmd: list[str],
    *,
    timeout: float,
    cwd: str,
    env: dict,
) -> subprocess.CompletedProcess:
    """subprocess.run 대체 — 타임아웃 시 프로세스 그룹 전체를 kill한다(#37).

    uv는 python을 exec로 치환하지 않고 자식으로 spawn해, 직속 자식만 죽이면 손자가
    고아로 남는다. start_new_session + os.killpg로 그룹째 정리한다.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()  # reap
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _kaggle_submit(
    submission_id: str,
    competition_id: str,
    competition_slug: str,
    attempt_id: str | None,
    message: str,
) -> None:
    """CSV 생성(캐시 우선) + kaggle 제출. 폴링은 /refresh 엔드포인트(DAG)가 담당.

    promote 시점에 캐싱된 CSV(store.s3_code.download_submission_csv)가
    있으면 그걸로 바로 업로드한다 — fit 없이 수 초. 캐시 미스(비승격 attempt 등)면
    기존대로 bin.submit 서브프로세스가 그 자리에서 fit한다(daemon 상주 ops-vm의
    아침 CPU 스파이크 원인이던 경로 — 캐시가 이걸 대체하는 게 이번 변경의 목적).
    """
    from store.db import connect
    from store.s3_code import download_submission_csv, upload_submission_csv

    def _update(fields: dict) -> None:
        c = connect(apply_schema=False)
        sets = ", ".join(f"{k} = %s" for k in fields)
        c.execute(
            f"update raw.kaggle_submissions set {sets} where submission_id = %s",
            list(fields.values()) + [submission_id],
        )
        c.close()

    _update({"status": "submitting", "checked_at": datetime.now(timezone.utc)})

    cached = download_submission_csv(competition_id, attempt_id) if attempt_id else None
    tmp_path: Path | None = None

    try:
        with _kaggle_home_env() as env:
            if env is None:
                _update({"status": "error", "error": "kaggle token unavailable", "checked_at": datetime.now(timezone.utc)})
                return
            if cached:
                tmp_path = Path(tempfile.gettempdir()) / f"rondo_submit_{submission_id}.csv"
                tmp_path.write_bytes(cached)
                result = _run_in_pgroup(
                    ["uv", "run", "kaggle", "competitions", "submit",
                     "-c", competition_id, "-f", str(tmp_path), "-m", message],
                    timeout=_SUBMIT_TIMEOUT_SEC, cwd=str(ROOT), env=env,
                )
                csv_path: str | None = str(tmp_path)
            else:
                cmd = [
                    "uv", "run", "python", "-m", "bin.submit",
                    "--competition", competition_slug,
                    "--submit", "--message", message,
                ]
                if attempt_id:
                    cmd += ["--attempt-id", attempt_id]
                result = _run_in_pgroup(
                    cmd, timeout=_SUBMIT_TIMEOUT_SEC, cwd=str(ROOT), env=env,
                )
                csv_path = None
                for line in result.stdout.splitlines():
                    if "submission saved:" in line:
                        csv_path = line.split("submission saved:", 1)[1].strip()
                        break
    except subprocess.TimeoutExpired:
        _update({
            "status": "error",
            "error": f"submit timed out ({_SUBMIT_TIMEOUT_SEC}s)",
            "checked_at": datetime.now(timezone.utc),
        })
        return
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        _update({"status": "error", "error": result.stderr[:2000], "checked_at": datetime.now(timezone.utc)})
        return

    # 캐시 미스로 여기서 직접 fit한 경우 — 결과를 캐시에 올려두면 같은 attempt의
    # 다음 제출부턴 fit 없이 히트한다(promote task가 미리 캐싱하지 못한 타이밍 갭 대비 안전망).
    if not cached and csv_path and attempt_id:
        try:
            upload_submission_csv(competition_id, attempt_id, Path(csv_path).read_bytes())
        except Exception as exc:
            print(f"  [submit] submission csv cache upload failed (non-fatal): {exc}")

    fields: dict = {"status": "submitted", "checked_at": datetime.now(timezone.utc)}
    if csv_path:
        fields["csv_path"] = csv_path
    _update(fields)


_kaggle_token_cache: dict = {}
_kaggle_token_lock = threading.Lock()


def _get_fresh_kaggle_token() -> str | None:
    """Return valid OAuth access token, refreshing via kagglesdk if needed.

    credentials.json is bind-mounted read-only in the daemon container.
    kagglesdk's refresh_access_token() always calls save() → OSError on ro mount.
    We call generate_access_token() directly (no save()) and cache the result.
    """
    with _kaggle_token_lock:
        now = datetime.now(timezone.utc)
        cached_token = _kaggle_token_cache.get("token")
        cached_exp = _kaggle_token_cache.get("expires_at")
        if cached_token and cached_exp and cached_exp > now + timedelta(minutes=5):
            return cached_token
        try:
            import kagglesdk.kaggle_creds as _kc
            from kagglesdk.kaggle_client import KaggleClient

            creds_path = Path.home() / ".kaggle" / "credentials.json"
            if not creds_path.exists():
                return None
            creds_data = json.loads(creds_path.read_text())
            client = KaggleClient()
            creds = _kc.KaggleCredentials(client=client, refresh_token=creds_data.get("refresh_token"))
            resp = creds.generate_access_token()
            _kaggle_token_cache["token"] = resp.token
            _kaggle_token_cache["expires_at"] = now + timedelta(seconds=resp.expires_in)
            return resp.token
        except Exception as exc:
            print(f"  [kaggle] token refresh failed: {exc}")
            return None


@contextlib.contextmanager
def _kaggle_home_env():
    """kaggle CLI가 OAuth credentials.json 대신 access_token 파일을 쓰도록 HOME 오버라이드.

    kagglesdk가 credentials.json의 save()를 항상 호출 → read-only mount에서 OSError.
    OAuth를 완전히 우회하기 위해: temp HOME dir에 .kaggle/access_token만 두고
    HOME 환경변수를 해당 경로로 설정 → kaggle CLI가 credentials.json을 찾지 못해
    OAuth 건너뜀 → access_token 파일로 직접 인증.
    """
    token = _get_fresh_kaggle_token()
    if not token:
        yield None
        return
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_home_"))
    try:
        (tmp / ".kaggle").mkdir()
        (tmp / ".kaggle" / "access_token").write_text(token)
        yield {**os.environ, "HOME": str(tmp)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _poll_kaggle_once(
    competition_id: str,
    message: str,
) -> tuple[str, float | None]:
    """kaggle submissions 1회 조회 → (status, lb_score).

    반환 status: 'complete' | 'error' | 'invalid' | 'pending'
    kaggle CLI 오류 시 'pending' 반환 (재시도 위임).
    """
    try:
        with _kaggle_home_env() as env:
            if env is None:
                print("  [poll/warn] kaggle token unavailable, skipping")
                return "pending", None
            poll = _run_in_pgroup(
                ["uv", "run", "kaggle", "competitions", "submissions",
                 "-c", competition_id, "--csv"],
                timeout=30, cwd=str(ROOT), env=env,
            )
        if poll.returncode != 0:
            print(f"  [poll/warn] kaggle CLI rc={poll.returncode}: {(poll.stderr or poll.stdout)[:200]!r}")
            return "pending", None

        # kaggle CLI may print Warning lines to stdout before CSV — skip them
        csv_lines = [l for l in poll.stdout.splitlines() if not l.startswith("Warning:")]
        reader = csv_module.DictReader(io.StringIO("\n".join(csv_lines)))
        matched = False
        for row in reader:
            if (row.get("description") or "").strip() != message:
                continue
            matched = True
            status = (row.get("status") or "").lower()
            if status.endswith("complete"):
                raw_score = row.get("publicScore") or row.get("public score") or ""
                lb_score: float | None = None
                try:
                    lb_score = float(raw_score)
                except (ValueError, TypeError):
                    pass
                return "complete", lb_score
            if status.endswith("error") or status.endswith("invalid"):
                return status.rsplit(".", 1)[-1], None
            return "pending", None  # 아직 채점 중
        if not matched:
            print(f"  [poll/warn] no kaggle row matched description={message!r}")
    except Exception as exc:
        print(f"  [poll/warn] exception: {exc}")
    return "pending", None


def _start_submission(
    conn: PgConn,
    competition_slug: str,
    competition_id: str,
    attempt_id: str | None,
    message: str,
) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        """
        insert into raw.kaggle_submissions
            (submission_id, competition_id, attempt_id, submitted_at, message, status)
        values (%s, %s, %s, %s, %s, 'queued')
        """,
        [sid, competition_id, attempt_id, datetime.now(timezone.utc), message],
    )
    threading.Thread(
        target=_kaggle_submit,
        args=(sid, competition_id, competition_slug, attempt_id, message),
        daemon=True,
    ).start()
    return sid


def _competition_id_to_slug() -> dict[str, str]:
    """config/competitions/*.py 스캔 → {competition_id: module_slug} 맵."""
    cached, hit = _cache.get("_comp_slug_map", ttl=3600)
    if hit:
        return cached  # type: ignore[return-value]
    result: dict[str, str] = {}
    comp_dir = ROOT / "config" / "competitions"
    for path in comp_dir.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"config.competitions.{path.stem}")
            cid = getattr(mod, "COMPETITION_ID", None)
            if cid:
                result[cid] = path.stem
        except Exception:
            continue
    _cache.set("_comp_slug_map", result)
    return result


def _best_attempt(conn: PgConn, competition_id: str) -> tuple[str, float] | None:
    row = conn.execute(
        """
        select a.attempt_id, a.cv_score
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = %s
          and a.cv_score is not null
          and a.error_trace is null
        order by c.metric_sign * a.cv_score desc, a.run_ts asc, a.attempt_id asc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    return (row[0], row[1]) if row else None


def _last_submitted_attempt(conn: PgConn, competition_id: str) -> str | None:
    """오래 안 풀린 'submitted'는 미확정으로 보고 제외한다(#52) — 안 그러면 재제출
    조건이 영원히 안 걸린다."""
    row = conn.execute(
        """
        select attempt_id from raw.kaggle_submissions
        where competition_id = %s
          and status not in ('error', 'invalid', 'timeout')
          and not (status = 'submitted' and checked_at < now() - interval '1 hour')
        order by submitted_at desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def create_app(conn: PgConn, state: DaemonState) -> FastAPI:
    app = FastAPI(title="reflexion-rondo", version="v1")

    @app.get("/api/heartbeat")
    def heartbeat():
        snap = state.snapshot()
        return {"status": "running" if snap["current_queue_id"] else "idle", **snap}

    @app.get("/api/health")
    def health():
        from bin.healthcheck import run_checks
        checks = run_checks()
        overall = "ok" if all(v["status"] != "fail" for v in checks.values()) else "degraded"
        return {"overall": overall, "checks": checks}

    @app.get("/api/competitions")
    def get_competitions():
        cached, hit = _cache.get("competitions", ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            "select competition_id, name, task_type, metric, metric_sign, start_ts from raw.competitions"
        ).fetchall()
        cols = ["competition_id", "name", "task_type", "metric", "metric_sign", "start_ts"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set("competitions", result)
        return result

    @app.get("/api/attempts")
    def get_attempts(
        competition: str | None = None,
        action_type: str | None = None,
        label: str | None = None,
        has_error: bool | None = None,
        stage: str | None = None,
        super_cycle_id: str | None = None,
        limit: int = 50,
    ):
        limit = min(limit, 500)
        cache_key = (
            f"attempts:{competition}:{action_type}:{label}:{has_error}:"
            f"{stage}:{super_cycle_id}:{limit}"
        )
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        conditions = []
        params: list = []
        if competition:
            conditions.append("a.competition_id = %s")
            params.append(competition)
        if action_type:
            conditions.append("a.action_type = %s")
            params.append(action_type)
        if label:
            conditions.append("a.label = %s")
            params.append(label)
        if has_error is not None:
            conditions.append("a.error_trace is not null" if has_error else "a.error_trace is null")
        if stage:
            conditions.append("a.stage = %s")
            params.append(stage)
        if super_cycle_id:
            conditions.append("a.super_cycle_id = %s")
            params.append(super_cycle_id)
        where = ("where " + " and ".join(conditions)) if conditions else ""
        rows = conn.execute(
            f"""
            select
                a.attempt_id, a.competition_id, a.run_ts, a.stage,
                a.hypothesis, a.action_type, a.cv_score, a.cv_fold_var,
                a.label, a.gain_vs_best, a.error_trace, a.duration_sec,
                a.retries, a.code_path,
                a.super_cycle_id, a.was_promoted, a.holdout_score,
                a.error_signature, a.retrieved_ids, a.reflection_ids,
                s.best_so_far
            from raw.attempts a
            left join score_progression s using (attempt_id)
            {where}
            order by a.run_ts desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "attempt_id", "competition_id", "run_ts", "stage",
            "hypothesis", "action_type", "cv_score", "cv_fold_var",
            "label", "gain_vs_best", "error_trace", "duration_sec",
            "retries", "code_path",
            "super_cycle_id", "was_promoted", "holdout_score",
            "error_signature", "retrieved_ids", "reflection_ids",
            "best_so_far",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/attempts/{attempt_id}")
    def get_attempt(attempt_id: str):
        row = conn.execute(
            """
            select
                attempt_id, competition_id, run_ts, stage,
                hypothesis, action_type, cv_score, cv_fold_var,
                label, gain_vs_best, error_trace, duration_sec,
                retries, code_path, reflection_ids, retrieval_scores,
                retrieved_ids, error_signature
            from raw.attempts
            where attempt_id = %s
            """,
            [attempt_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="attempt not found")
        cols = [
            "attempt_id", "competition_id", "run_ts", "stage",
            "hypothesis", "action_type", "cv_score", "cv_fold_var",
            "label", "gain_vs_best", "error_trace", "duration_sec",
            "retries", "code_path", "reflection_ids", "retrieval_scores",
            "retrieved_ids", "error_signature",
        ]
        return dict(zip(cols, row))

    _LESSON_SORT_COLUMNS = {
        "created_at": "r.created_at",
        "avg_gain": "avg_gain",
        "times_applied": "times_applied",
        "jumps": "jumps",
    }

    @app.get("/api/lessons")
    def get_lessons(
        competition: str | None = None,
        generality: str | None = None,
        lesson_type: str | None = None,
        archived: bool | None = None,
        sort: str = "created_at",
        limit: int = 50,
    ):
        limit = min(limit, 500)
        cache_key = (
            f"lessons:{competition}:{generality}:{lesson_type}:{archived}:{sort}:{limit}"
        )
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        sort_col = _LESSON_SORT_COLUMNS.get(sort, "r.created_at")
        conditions = ["r.archived = false" if archived is None else "r.archived = %s"]
        params: list = [] if archived is None else [archived]
        if competition:
            conditions.append("(r.competition_id = %s or r.generality in ('L2_class', 'L3_general'))")
            params.append(competition)
        if generality:
            conditions.append("r.generality = %s")
            params.append(generality)
        if lesson_type:
            conditions.append("r.lesson_type = %s")
            params.append(lesson_type)
        where = "where " + " and ".join(conditions)
        rows = conn.execute(
            f"""
            select
                r.reflection_id, r.created_at, r.competition_id,
                r.embedded_text, r.full_lesson, r.generality,
                r.label, r.gain_vs_best,
                r.lesson_type, r.reflector_label, r.archived,
                coalesce(i.times_applied, 0) as times_applied,
                coalesce(i.avg_gain, 0.0) as avg_gain,
                coalesce(i.jumps, 0) as jumps,
                coalesce(i.best_jump, 0.0) as best_jump,
                (select count(*) from raw.attempts a2
                 where r.reflection_id = any(a2.retrieved_ids)) as times_retrieved
            from raw.reflections r
            left join reflection_impact i using (reflection_id)
            {where}
            order by {sort_col} desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "reflection_id", "created_at", "competition_id",
            "embedded_text", "full_lesson", "generality",
            "label", "gain_vs_best",
            "lesson_type", "reflector_label", "archived",
            "times_applied", "avg_gain", "jumps", "best_jump", "times_retrieved",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/cold-start")
    def get_cold_start(competition: str | None = None, limit: int = 200):
        limit = min(limit, 1000)
        cache_key = f"cold_start:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        where = "where competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select competition_id, attempt_no, run_ts, stage, cv_score, best_so_far
            from cold_start_progression
            {where}
            order by competition_id, attempt_no
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = ["competition_id", "attempt_no", "run_ts", "stage", "cv_score", "best_so_far"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    # ---- 관측 (#11/#67) — 건강질문 직결 8개, 데이터는 store/schema.sql 파생 뷰 ----

    @app.get("/api/score/timeline")
    def get_score_timeline(competition: str, limit: int = 2000):
        limit = min(limit, 5000)
        cache_key = f"score_timeline:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        rows = conn.execute(
            """
            select s.attempt_no, s.run_ts, s.cv_score, s.best_so_far, s.label,
                   (s.label = 'jump') as is_jump, h.holdout_score
            from score_progression s
            left join holdout_cv_gap_trend h using (attempt_id)
            where s.competition_id = %s
            order by s.attempt_no
            limit %s
            """,
            [competition, limit],
        ).fetchall()
        cols = ["attempt_no", "run_ts", "cv_score", "best_so_far", "label", "is_jump", "holdout_score"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/bandit/posteriors")
    def get_bandit_posteriors(competition: str):
        cache_key = f"bandit_posteriors:{competition}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select action_type, alpha, beta, posterior_mean, net_evidence,
                   picks, last_picked_ts, weighted_success, calibration_gap
            from bandit_calibration
            where scope = 'local' and scope_key = %s
            order by posterior_mean desc
            """,
            [competition],
        ).fetchall()
        cols = [
            "action_type", "alpha", "beta", "posterior_mean", "net_evidence",
            "picks", "last_picked_ts", "weighted_success", "calibration_gap",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/lessons/funnel")
    def get_lessons_funnel(competition: str):
        cache_key = f"lessons_funnel:{competition}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        row = conn.execute(
            """
            select competition_id, written, total_attempts, retrieved, cited, positive_gain,
                   retrieve_rate, cite_rate, gain_rate, retrieved_precise
            from lesson_funnel
            where competition_id = %s
            """,
            [competition],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="no funnel data for competition")
        cols = [
            "competition_id", "written", "total_attempts", "retrieved", "cited", "positive_gain",
            "retrieve_rate", "cite_rate", "gain_rate", "retrieved_precise",
        ]
        result = dict(zip(cols, row))
        _cache.set(cache_key, result)
        return result

    @app.get("/api/lessons/dead")
    def get_lessons_dead(competition: str, limit: int = 200):
        limit = min(limit, 1000)
        cache_key = f"lessons_dead:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        rows = conn.execute(
            """
            select reflection_id, competition_id, lesson_type, generality, created_at,
                   times_cited, avg_gain, reason
            from lesson_dead
            where competition_id = %s
            order by times_cited asc, created_at asc
            limit %s
            """,
            [competition, limit],
        ).fetchall()
        cols = [
            "reflection_id", "competition_id", "lesson_type", "generality", "created_at",
            "times_cited", "avg_gain", "reason",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/lessons/duplicates")
    def get_lessons_duplicates(competition: str, threshold: float = 0.95, limit: int = 200):
        limit = min(limit, 1000)
        cache_key = f"lessons_duplicates:{competition}:{threshold}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=120)
        if hit:
            return cached
        rows = conn.execute(
            """
            select competition_id, reflection_id_a, reflection_id_b, cos_sim
            from lesson_duplicates
            where competition_id = %s and cos_sim >= %s
            order by cos_sim desc
            limit %s
            """,
            [competition, threshold, limit],
        ).fetchall()
        cols = ["competition_id", "reflection_id_a", "reflection_id_b", "cos_sim"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/errors/signatures")
    def get_error_signatures(competition: str, limit: int = 100):
        limit = min(limit, 500)
        cache_key = f"error_signatures:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select action_type, error_signature, total, first_seen, last_seen,
                   pitfall_active, occurrences_after_active, has_avoid_lesson
            from error_recurrence
            where competition_id = %s
            order by total desc
            limit %s
            """,
            [competition, limit],
        ).fetchall()
        cols = [
            "action_type", "error_signature", "total", "first_seen", "last_seen",
            "pitfall_active", "occurrences_after_active", "has_avoid_lesson",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/transfer/matrix")
    def get_transfer_matrix():
        cached, hit = _cache.get("transfer_matrix", ttl=120)
        if hit:
            return cached
        rows = conn.execute(
            "select source_comp, target_comp, citations from transfer_matrix order by citations desc"
        ).fetchall()
        cols = ["source_comp", "target_comp", "citations"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set("transfer_matrix", result)
        return result

    @app.get("/api/reflexion-health")
    def get_reflexion_health(competition: str):
        """#11 §6 건강 신호등 4칸(T5). /api/health(의존성 헬스체크)와 경로 충돌 피하려 개명."""
        cache_key = f"reflexion_health:{competition}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached

        funnel = conn.execute(
            "select cite_rate, retrieved_precise from lesson_funnel where competition_id = %s",
            [competition],
        ).fetchone()
        cite_rate = float(funnel[0]) if funnel and funnel[0] is not None else 0.0
        citation_rate_approx = not bool(funnel[1]) if funnel else True

        label_rows = conn.execute(
            """
            select label from raw.attempts
            where competition_id = %s and stage = 'reflexion'
            order by run_ts desc limit 10
            """,
            [competition],
        ).fetchall()
        jumps_last10 = sum(1 for (label,) in label_rows if label == "jump")

        bandit_rows = conn.execute(
            "select posterior_mean, calibration_gap from bandit_calibration where scope_key = %s",
            [competition],
        ).fetchall()
        posteriors = [float(r[0]) for r in bandit_rows if r[0] is not None]
        gaps = [float(r[1]) for r in bandit_rows if r[1] is not None]
        posterior_spread = (max(posteriors) - min(posteriors)) if posteriors else 0.0
        calibration_gap = max(gaps) if gaps else 0.0

        error_rows = conn.execute(
            """
            select total, occurrences_after_active, pitfall_active
            from error_recurrence where competition_id = %s
            """,
            [competition],
        ).fetchall()
        active = [r for r in error_rows if r[2]]
        active_total = sum(r[0] for r in active)
        repeat_after_pitfall_rate = (
            sum(r[1] for r in active) / active_total if active_total else 0.0
        )

        # 최근/이전 절반 에러율 비교로 방향만 판정 — 신호등 용도라 선형회귀는 과함.
        recent_labels = conn.execute(
            """
            select (label = 'error') as is_error from raw.attempts
            where competition_id = %s and stage = 'reflexion'
            order by run_ts desc limit 200
            """,
            [competition],
        ).fetchall()
        error_rate_slope = 0
        if len(recent_labels) >= 20:
            mid = len(recent_labels) // 2
            recent = [e for (e,) in recent_labels[:mid]]
            earlier = [e for (e,) in recent_labels[mid:]]
            recent_rate = sum(recent) / len(recent)
            earlier_rate = sum(earlier) / len(earlier)
            error_rate_slope = -1 if recent_rate < earlier_rate else (1 if recent_rate > earlier_rate else 0)

        stagnation = detect_stagnation(conn, competition)
        action_coverage = len(ACTION_TYPES) - len(stagnation.underused_actions)

        result = {
            "competition_id": competition,
            "accumulation": {
                "status": _traffic_light(
                    green=(cite_rate >= 0.30 and jumps_last10 >= 1),
                    red=(cite_rate < 0.10 or (jumps_last10 == 0 and stagnation.is_stagnant)),
                ),
                "jumps_last10": jumps_last10,
                "citation_rate": round(cite_rate, 4),
                "citation_rate_approx": citation_rate_approx,
            },
            "bandit": {
                "status": _traffic_light(
                    green=(posterior_spread >= 0.15 and calibration_gap <= 0.15),
                    red=(posterior_spread < 0.05 or calibration_gap > 0.30),
                ),
                "posterior_spread": round(posterior_spread, 4),
                "calibration_gap": round(calibration_gap, 4),
            },
            "antipattern": {
                "status": _traffic_light(
                    green=(error_rate_slope < 0 and repeat_after_pitfall_rate <= 0.20),
                    red=(repeat_after_pitfall_rate > 0.50),
                ),
                "error_rate_slope": error_rate_slope,
                "repeat_after_pitfall_rate": round(repeat_after_pitfall_rate, 4),
            },
            "exploration": {
                "status": _traffic_light(
                    green=(action_coverage >= 3),
                    red=(stagnation.is_stagnant and action_coverage <= 1),
                ),
                "action_coverage": action_coverage,
                "actions_total": len(ACTION_TYPES),
                "is_stagnant": stagnation.is_stagnant,
            },
        }
        _cache.set(cache_key, result)
        return result

    # ---- 관측 (#11/#67) — 상위 레이어 T1/T2/T4 ----
    # best_cv 계산은 cycle/run.py:_prev_best와 동일 로직: 확정 pipelines 우선,
    # 없으면(cold-start) attempts max로 폴백.
    _BEST_CV_SQL = """
        coalesce(
            (select max(p.cv_score * cc.metric_sign) * cc.metric_sign
             from raw.pipelines p where p.competition_id = cc.competition_id and p.cv_score is not null),
            (select max(a.cv_score * cc.metric_sign) * cc.metric_sign
             from raw.attempts a where a.competition_id = cc.competition_id and a.cv_score is not null)
        )
    """

    @app.get("/api/competitions/summary")
    def get_competitions_summary():
        cached, hit = _cache.get("competitions_summary", ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            f"""
            select
                cc.competition_id, cc.name, cc.task_type, cc.metric,
                {_BEST_CV_SQL} as best_cv,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id) as n_attempts,
                (select count(*) from raw.pipelines p where p.competition_id = cc.competition_id) as n_pipelines,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id and a.label = 'jump') as n_jumps,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id and a.label = 'error') as n_errors,
                (select max(a.run_ts) from raw.attempts a where a.competition_id = cc.competition_id) as last_run_ts,
                (select q.status from raw.cycle_queue q where q.competition = cc.competition_id
                 order by q.created_at desc limit 1) as queue_status
            from raw.competitions cc
            order by cc.competition_id
            """
        ).fetchall()
        cols = [
            "competition_id", "name", "task_type", "metric", "best_cv",
            "n_attempts", "n_pipelines", "n_jumps", "n_errors", "last_run_ts", "queue_status",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set("competitions_summary", result)
        return result

    @app.get("/api/score")
    def get_score(competition: str):
        cache_key = f"score:{competition}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        row = conn.execute(
            f"""
            select
                {_BEST_CV_SQL} as best_cv,
                (select max(a.holdout_score) from raw.attempts a
                 where a.competition_id = cc.competition_id and a.was_promoted is true) as best_holdout,
                (select avg(cc.metric_sign * (a.cv_score - a.holdout_score)) from raw.attempts a
                 where a.competition_id = cc.competition_id and a.was_promoted is true
                   and a.holdout_score is not null) as cv_minus_holdout,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id) as n_attempts,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id and a.label = 'jump') as n_jumps,
                (select count(*) from raw.attempts a where a.competition_id = cc.competition_id and a.label = 'error') as n_errors,
                (select round(count(*) filter (where a.label = 'error')::numeric / nullif(count(*), 0), 4)
                 from raw.attempts a where a.competition_id = cc.competition_id and a.stage = 'reflexion') as error_rate,
                (select attempt_no from score_progression
                 where competition_id = cc.competition_id and label = 'jump'
                 order by attempt_no desc limit 1) as last_jump_attempt_no
            from raw.competitions cc
            where cc.competition_id = %s
            """,
            [competition],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="competition not found")
        cols = [
            "best_cv", "best_holdout", "cv_minus_holdout", "n_attempts",
            "n_jumps", "n_errors", "error_rate", "last_jump_attempt_no",
        ]
        result = dict(zip(cols, row))
        result["competition_id"] = competition
        stagnation = detect_stagnation(conn, competition)
        result["is_stagnant"] = stagnation.is_stagnant
        result["stagnant_for"] = stagnation.stagnant_for
        _cache.set(cache_key, result)
        return result

    @app.get("/api/best-strategy")
    def get_best_strategy(competition: str):
        cache_key = f"best_strategy:{competition}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        best = conn.execute(
            """
            select p.pipeline_id, p.cv_score, a.holdout_score
            from raw.pipelines p
            join raw.competitions c using (competition_id)
            join raw.attempts a using (attempt_id)
            where p.competition_id = %s and p.cv_score is not null
            order by c.metric_sign * p.cv_score desc
            limit 1
            """,
            [competition],
        ).fetchone()
        if not best:
            raise HTTPException(status_code=404, detail="no promoted pipeline for competition")
        pipeline_id, best_cv, best_holdout = best
        n_promotions, last_promoted_ts = conn.execute(
            "select count(*), max(a.run_ts) from raw.pipelines p join raw.attempts a using (attempt_id) "
            "where p.competition_id = %s",
            [competition],
        ).fetchone()
        contributing = conn.execute(
            """
            select a.action_type, count(*) as promotions, sum(a.gain_vs_best) as total_gain
            from raw.pipelines p join raw.attempts a using (attempt_id)
            where p.competition_id = %s
            group by a.action_type order by promotions desc
            """,
            [competition],
        ).fetchall()
        cited_ids = [
            r[0] for r in conn.execute(
                """
                select distinct rid
                from raw.pipelines p join raw.attempts a using (attempt_id), unnest(a.reflection_ids) rid
                where p.competition_id = %s
                """,
                [competition],
            ).fetchall()
        ]
        cited_lessons = []
        if cited_ids:
            cited_lessons = [
                {"reflection_id": r[0], "lesson_type": r[1], "generality": r[2],
                 "embedded_text": r[3], "avg_gain": r[4]}
                for r in conn.execute(
                    """
                    select r.reflection_id, r.lesson_type, r.generality, r.embedded_text,
                           coalesce(i.avg_gain, 0.0)
                    from raw.reflections r left join reflection_impact i using (reflection_id)
                    where r.reflection_id = any(%s::text[])
                    """,
                    [cited_ids],
                ).fetchall()
            ]
        result = {
            "competition_id": competition,
            "pipeline_id": pipeline_id,
            "best_cv": best_cv,
            "best_holdout": best_holdout,
            "n_promotions": n_promotions,
            "last_promoted_ts": last_promoted_ts,
            "contributing_actions": [
                {"action_type": r[0], "promotions": r[1], "total_gain": r[2]} for r in contributing
            ],
            "cited_lessons": cited_lessons,
        }
        _cache.set(cache_key, result)
        return result

    # ---- 관측 (#11/#67) — 밴딧 리플레이 B2b/B2c ----

    @app.get("/api/bandit/timeline")
    def get_bandit_timeline(competition: str):
        cache_key = f"bandit_timeline:{competition}"
        cached, hit = _cache.get(cache_key, ttl=120)
        if hit:
            return cached
        result = _replay_bandit_timeline(conn, competition)
        _cache.set(cache_key, result)
        return result

    @app.get("/api/bandit/selection")
    def get_bandit_selection(competition: str):
        cache_key = f"bandit_selection:{competition}"
        cached, hit = _cache.get(cache_key, ttl=120)
        if hit:
            return cached
        timeline = _replay_bandit_timeline(conn, competition)
        sc_rows = conn.execute(
            """
            select super_cycle_id, created_at, assigned_actions
            from raw.super_cycle_context
            where competition_id = %s and assigned_actions is not null
            order by created_at
            """,
            [competition],
        ).fetchall()
        result = []
        for super_cycle_id, created_at, assigned_actions in sc_rows:
            assigned = (
                assigned_actions if isinstance(assigned_actions, list)
                else json.loads(assigned_actions)
            )
            ranked = _posterior_rank_at(timeline, created_at)
            top3_assigned = set(assigned[:3])
            top3_ranked = set(ranked[:3])
            concordance = (
                len(top3_assigned & top3_ranked) / len(top3_assigned) if top3_assigned else 0.0
            )
            result.append({
                "super_cycle_id": super_cycle_id,
                "run_ts": created_at,
                "assigned_actions": assigned,
                "posterior_rank_at_time": ranked,
                "concordance": round(concordance, 4),
            })
        _cache.set(cache_key, result)
        return result

    # ---- 관측 (#11/#67) — 교훈 generality-mix + 에러 rate-timeline/repeat-offenders ----

    @app.get("/api/lessons/generality-mix")
    def get_generality_mix(competition: str, bucket: str = "week"):
        if bucket not in ("day", "week"):
            raise HTTPException(status_code=422, detail="bucket must be 'day' or 'week'")
        cache_key = f"generality_mix:{competition}:{bucket}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select
                date_trunc(%s, created_at) as bucket_ts,
                count(*) filter (where generality = 'L1_local') as l1_local,
                count(*) filter (where generality = 'L2_class') as l2_class,
                count(*) filter (where generality = 'L3_general') as l3_general
            from raw.reflections
            where competition_id = %s or generality in ('L2_class', 'L3_general')
            group by bucket_ts
            order by bucket_ts
            """,
            [bucket, competition],
        ).fetchall()
        cols = ["bucket_ts", "l1_local", "l2_class", "l3_general"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/errors/rate-timeline")
    def get_error_rate_timeline(competition: str):
        cache_key = f"error_rate_timeline:{competition}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select
                super_cycle_id, min(run_ts) as run_ts, count(*) as n_attempts,
                count(*) filter (where label = 'error') as n_errors,
                round(count(*) filter (where label = 'error')::numeric / count(*), 4) as error_rate
            from raw.attempts
            where competition_id = %s and super_cycle_id is not null
            group by super_cycle_id
            order by run_ts
            """,
            [competition],
        ).fetchall()
        cols = ["super_cycle_id", "run_ts", "n_attempts", "n_errors", "error_rate"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/errors/repeat-offenders")
    def get_repeat_offenders(competition: str, limit: int = 100):
        limit = min(limit, 500)
        cache_key = f"repeat_offenders:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select error_signature, action_type, total, occurrences_after_active, last_seen
            from error_recurrence
            where competition_id = %s and occurrences_after_active > 0
            order by occurrences_after_active desc
            limit %s
            """,
            [competition, limit],
        ).fetchall()
        cols = ["error_signature", "action_type", "total", "after_active", "last_seen"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    # ---- 관측 (#11/#67) — 승격 타임라인 + 전이 리더보드/fp-distance ----

    @app.get("/api/promotions")
    def get_promotions(competition: str):
        cache_key = f"promotions:{competition}"
        cached, hit = _cache.get(cache_key, ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            """
            select
                a.attempt_id, a.run_ts, a.action_type, a.gain_vs_best, a.cv_score,
                a.holdout_score, a.hypothesis,
                max(cc.metric_sign * a.cv_score) over (
                    order by a.run_ts rows between unbounded preceding and current row
                ) * cc.metric_sign as cumulative_best
            from raw.pipelines p
            join raw.attempts a using (attempt_id)
            join raw.competitions cc on cc.competition_id = p.competition_id
            where p.competition_id = %s
            order by a.run_ts
            """,
            [competition],
        ).fetchall()
        cols = [
            "attempt_id", "run_ts", "action_type", "gain_vs_best", "cv_score",
            "holdout_score", "hypothesis", "cumulative_best",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/transfer/lessons")
    def get_transfer_lessons(limit: int = 100):
        limit = min(limit, 500)
        cache_key = f"transfer_lessons:{limit}"
        cached, hit = _cache.get(cache_key, ttl=120)
        if hit:
            return cached
        rows = conn.execute(
            """
            select
                r.reflection_id, r.generality, r.embedded_text, r.competition_id as source_comp,
                count(distinct a.competition_id)
                    filter (where a.competition_id <> r.competition_id) as reused_in_comps,
                count(*) as total_citations,
                coalesce(avg(a.gain_vs_best_relative), 0.0) as avg_gain
            from raw.reflections r
            join raw.attempts a on r.reflection_id = any(a.reflection_ids)
            where r.generality in ('L2_class', 'L3_general') and r.archived = false
            group by r.reflection_id, r.generality, r.embedded_text, r.competition_id
            having count(distinct a.competition_id) filter (where a.competition_id <> r.competition_id) > 0
            order by reused_in_comps desc, total_citations desc
            limit %s
            """,
            [limit],
        ).fetchall()
        cols = [
            "reflection_id", "generality", "embedded_text", "source_comp",
            "reused_in_comps", "total_citations", "avg_gain",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/transfer/fp-distance")
    def get_fp_distance():
        """대회별 최근접 선행 대회 + fingerprint 거리. memory/transfer.py:_fp_distance
        재사용(순수 함수) + export_cold_start_summary(bin/export_results.py)와 동일 SQL로
        warm_start_ratio 병합."""
        cached, hit = _cache.get("fp_distance", ttl=300)
        if hit:
            return cached
        comps = conn.execute("select competition_id, fingerprint from raw.competitions").fetchall()
        fps = [
            (cid, fp if isinstance(fp, dict) else json.loads(fp))
            for cid, fp in comps if fp
        ]
        cold_start = conn.execute(
            """
            select
                cc.competition_id,
                max(case when p.stage = 'bootstrap' then p.best_so_far end) as bootstrap_best,
                max(p.best_so_far) as overall_best
            from cold_start_progression p
            join raw.competitions cc using (competition_id)
            group by cc.competition_id
            """
        ).fetchall()
        cold_start_map = {r[0]: (r[1], r[2]) for r in cold_start}
        result = []
        for cid, fp in fps:
            nearest_id, nearest_dist = None, None
            for oid, ofp in fps:
                if oid == cid:
                    continue
                d = _fp_distance(fp, ofp)
                if nearest_dist is None or d < nearest_dist:
                    nearest_id, nearest_dist = oid, d
            bootstrap_best, overall_best = cold_start_map.get(cid, (None, None))
            warm_start_ratio = (
                round(bootstrap_best / overall_best, 4)
                if bootstrap_best is not None and overall_best not in (None, 0)
                else None
            )
            result.append({
                "competition_id": cid,
                "nearest_prior_comp": nearest_id,
                "fp_distance": round(nearest_dist, 4) if nearest_dist is not None else None,
                "warm_start_ratio": warm_start_ratio,
                "bootstrap_best": bootstrap_best,
                "overall_best": overall_best,
            })
        _cache.set("fp_distance", result)
        return result

    @app.get("/api/queue")
    def get_queue(status: str | None = None, limit: int = 100):
        limit = min(limit, 500)
        where = "where status = %s" if status else ""
        params = [status] if status else []
        rows = conn.execute(
            f"""
            select queue_id, competition, stage, n_cycles, priority,
                   status, created_at, started_at, ended_at,
                   cycles_done, latest_score, error
            from raw.cycle_queue
            {where}
            order by priority desc, created_at asc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "queue_id", "competition", "stage", "n_cycles", "priority",
            "status", "created_at", "started_at", "ended_at",
            "cycles_done", "latest_score", "error",
        ]
        return [dict(zip(cols, r)) for r in rows]

    @app.post("/api/queue", status_code=201)
    def enqueue(body: EnqueueRequest):
        try:
            importlib.import_module(f"config.competitions.{body.competition}")
        except ModuleNotFoundError:
            raise HTTPException(status_code=422, detail=f"unknown competition: {body.competition}")
        qid = str(uuid.uuid4())
        conn.execute(
            """
            insert into raw.cycle_queue
                (queue_id, competition, stage, n_cycles, priority, status, created_at)
            values (%s, %s, %s, %s, %s, 'pending', %s)
            """,
            [qid, body.competition, body.stage, body.n_cycles,
             body.priority, datetime.now(timezone.utc)],
        )
        return {"queue_id": qid}

    @app.patch("/api/queue/{queue_id}")
    def patch_queue(queue_id: str, body: QueuePatchRequest):
        row = conn.execute(
            "select status from raw.cycle_queue where queue_id = %s",
            [queue_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="queue item not found")

        sets: list[str] = []
        vals: list = []
        if body.priority is not None:
            sets.append("priority = %s")
            vals.append(body.priority)
        if body.status == "cancelled":
            if row[0] not in ("pending", "running"):
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot cancel item with status '{row[0]}'",
                )
            sets.append("status = %s")
            vals.append("cancelled")

        if not sets:
            raise HTTPException(status_code=422, detail="nothing to update")

        vals.append(queue_id)
        conn.execute(
            f"update raw.cycle_queue set {', '.join(sets)} where queue_id = %s",
            vals,
        )
        return {"queue_id": queue_id, "updated": True}

    @app.post("/api/submissions", status_code=201)
    def submit(body: SubmitRequest):
        try:
            comp = importlib.import_module(f"config.competitions.{body.competition}")
        except ModuleNotFoundError:
            raise HTTPException(status_code=422, detail=f"unknown competition: {body.competition}")

        msg = body.message or f"rondo attempt={body.attempt_id or 'best'}"
        sid = _start_submission(conn, body.competition, comp.COMPETITION_ID, body.attempt_id, msg)
        return {"submission_id": sid, "status": "queued"}

    @app.post("/api/submissions/auto", status_code=200)
    def auto_submit(body: AutoSubmitRequest):
        slug_map = _competition_id_to_slug()

        active_rows = conn.execute(
            """
            select distinct competition_id from raw.attempts
            where run_ts >= now() - make_interval(hours => %s)
              and cv_score is not null
              and error_trace is null
            """,
            [body.window_hours],
        ).fetchall()

        submitted = []
        skipped = []

        for (competition_id,) in active_rows:
            slug = slug_map.get(competition_id)
            if not slug:
                skipped.append({"competition": competition_id, "reason": "no config"})
                continue

            best = _best_attempt(conn, competition_id)
            if not best:
                skipped.append({"competition": competition_id, "reason": "no valid attempt"})
                continue
            best_attempt_id, best_cv = best

            last = _last_submitted_attempt(conn, competition_id)
            if last == best_attempt_id:
                skipped.append({"competition": competition_id, "reason": "best unchanged"})
                continue

            msg = f"auto cv={best_cv:.5f} attempt={best_attempt_id[:8]}"
            sid = _start_submission(conn, slug, competition_id, best_attempt_id, msg)
            submitted.append({
                "competition": competition_id,
                "slug": slug,
                "attempt_id": best_attempt_id,
                "submission_id": sid,
            })

        return {"submitted": submitted, "skipped": skipped}

    @app.get("/api/submissions")
    def get_submissions(competition: str | None = None, limit: int = 50):
        limit = min(limit, 200)
        where = "where competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select submission_id, competition_id, attempt_id, submitted_at,
                   message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            {where}
            order by submitted_at desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    @app.get("/api/submissions/{submission_id}")
    def get_submission(submission_id: str):
        row = conn.execute(
            """
            select submission_id, competition_id, attempt_id, submitted_at,
                   message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            where submission_id = %s
            """,
            [submission_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="submission not found")
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        return dict(zip(cols, row))

    @app.post("/api/submissions/{submission_id}/refresh")
    def refresh_submission(submission_id: str):
        row = conn.execute(
            """
            select submission_id, competition_id, attempt_id,
                   submitted_at, message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            where submission_id = %s
            """,
            [submission_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="submission not found")
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        rec = dict(zip(cols, row))

        if rec["status"] in _TERMINAL:
            return rec

        kaggle_status, lb_score = _poll_kaggle_once(rec["competition_id"], rec["message"])
        if kaggle_status == "pending":
            conn.execute(
                "update raw.kaggle_submissions set checked_at = %s where submission_id = %s",
                [datetime.now(timezone.utc), submission_id],
            )
            rec["checked_at"] = datetime.now(timezone.utc)
            return rec

        fields: dict = {"status": kaggle_status, "checked_at": datetime.now(timezone.utc)}
        if kaggle_status == "complete" and lb_score is not None:
            fields["lb_score"] = lb_score
        elif kaggle_status in ("error", "invalid"):
            fields["error"] = f"kaggle: {kaggle_status}"

        sets = ", ".join(f"{k} = %s" for k in fields)
        conn.execute(
            f"update raw.kaggle_submissions set {sets} where submission_id = %s",
            list(fields.values()) + [submission_id],
        )

        if kaggle_status == "complete" and lb_score is not None and rec.get("attempt_id"):
            conn.execute(
                "update raw.attempts set lb_score = %s where attempt_id like %s and lb_score is null",
                [lb_score, f"{rec['attempt_id']}%"],
            )

        rec.update(fields)
        return rec

    return app
