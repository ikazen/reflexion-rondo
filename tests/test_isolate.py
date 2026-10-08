"""runtime/isolate.py 리소스 상한 회귀 테스트.

mac-server-big(당시 Colima VM 8GiB 고정)의 이론상 오버서브스크립션을 막으려고
RLIMIT_AS를 6GiB→1.5GiB로 낮췄으나, RLIMIT_AS는 물리 RSS가 아니라 가상 주소공간(VSZ)
상한이라 numpy/scipy/sklearn 등 라이브러리를 import하는 것만으로도 부족해 신규 대회
부트스트랩 전체가 실패하는 회귀를 냈다 — 물리 메모리가 남는 worker-vm에서도 실패.

이후 mac-server Colima VM을 8→16GiB로 증설하고(실측 최대 동시성도 3이지 4가 아님을
확인), RLIMIT_AS를 원래 값 6GiB로 복원했다.

RLIMIT_CPU는 과거 soft==hard==900으로 걸었으나, 리눅스가 hard를 먼저 검사해
SIGXCPU 없이 곧장 SIGKILL(rc=-9)로 죽여 OOM killer 사망과 구분이 안 됐다
(2026-08 실측: 계산의 40%, 전부 재시도까지 태워 attempt당 최대 16분). 지금은
eval_isolated의 폴링 루프가 CPU 시간을 직접 감시해서 명시적 원인으로 선제
kill하고, RLIMIT_CPU는 폴링이 놓쳤을 때만 발동하는 soft<hard 백스톱으로 강등했다.

os.unshare(CLONE_NEWNET)는 CAP_SYS_ADMIN을 요구하고 테스트 프로세스 자체의
네트워크 namespace에 영향을 줄 수 있어 건드리지 않는다 — _set_resource_limits()는
그 로직과 분리돼 있어 안전하게 단위 테스트 가능하다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.isolate import (
    _CPU_BACKSTOP_HARD_MARGIN_SECS,
    _CPU_BACKSTOP_SOFT_MARGIN_SECS,
    _DEFAULT_MEM_LIMIT_BYTES,
    _make_preexec,
    _set_resource_limits,
)

_EXPECTED_DEFAULT = 6 * 1024 ** 3


def test_default_mem_limit_is_6_gib() -> None:
    assert _DEFAULT_MEM_LIMIT_BYTES == _EXPECTED_DEFAULT


def test_set_resource_limits_uses_new_default_when_no_override() -> None:
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=900)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (_EXPECTED_DEFAULT, _EXPECTED_DEFAULT)
    assert calls[resource.RLIMIT_CPU] == (
        900 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 900 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


def test_set_resource_limits_respects_mem_env_override() -> None:
    override_bytes = str(3 * 1024 ** 3)
    with patch.dict("os.environ", {"EVAL_MEM_LIMIT_BYTES": override_bytes}), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=900)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (3 * 1024 ** 3, 3 * 1024 ** 3)


def test_set_resource_limits_cpu_backstop_tracks_budget_argument() -> None:
    """RLIMIT_CPU 백스톱은 env가 아니라 호출자가 넘긴 cpu_budget에서 파생된다 —
    eval_isolated가 attempt 단위로 남은 예산을 계산해 매 재시도마다 다르게 넘기므로."""
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=200)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_CPU] == (
        200 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 200 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


def test_make_preexec_returns_callable_that_applies_budget() -> None:
    with patch("runtime.isolate._resource.setrlimit") as mock_setrlimit, \
         patch("runtime.isolate._HAVE_NEWNET", False):
        preexec = _make_preexec(cpu_budget=900)
        assert preexec is not None
        preexec()

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_CPU] == (
        900 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 900 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


#
# eval_isolated는 내부적으로 subprocess.run 대신 Popen + RSS 폴링 루프를 쓴다
# (runner exited rc=-9 OOM kill이 평균 773초를 태우고서야 죽는 것을 워치독으로
# 선제 차단하기 위함 — 2026-08 처리량 진단). 그래서 아래 fake는 subprocess.run이
# 아니라 Popen을 대체하고, poll 루프가 기대하는 pid/wait()/kill() 인터페이스를
# 갖춘다.

class _FakePopen:
    """subprocess.Popen 대역 — pid는 실존하지 않아 _read_rss_bytes가 항상 None을
    반환하므로(실제 프로세스 없음), RSS 워치독 분기를 타지 않고 wait()에서 바로
    종료한다."""

    def __init__(self, cmd, **kwargs):
        self.pid = 2 ** 30  # 실존 가능성이 사실상 0인 pid
        self.returncode = 0
        self._on_init(cmd)

    def _on_init(self, cmd) -> None:
        pass

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        pass


def test_eval_isolated_passes_through_gain_vs_best_relative() -> None:
    """subprocess(runner.py)가 쓴 output.json의 gain_vs_best_relative가 IsolatedResult로
    그대로 전달되는지 확인 — metric 스케일 정규화 필드가 격리 경계를 넘어야 한다."""
    import json as _json

    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def _on_init(self, cmd) -> None:
            tmpdir = cmd[2]
            (Path(tmpdir) / "output.json").write_text(_json.dumps({
                "cv_score": 0.9, "cv_fold_var": 0.01, "fold_scores": [0.89, 0.9, 0.91],
                "label": "jump", "gain_vs_best": 0.05, "gain_vs_best_relative": 0.06,
                "error_trace": None,
            }))

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
        )
    assert result.gain_vs_best_relative == 0.06
    assert result.peak_rss_bytes is None  # pid가 실존하지 않아 RSS를 못 읽음
    assert result.peak_cpu_sec is None  # 마찬가지로 CPU 시간도 못 읽음


def test_eval_isolated_kills_on_rss_over_limit() -> None:
    """peak RSS가 EVAL_RSS_LIMIT_BYTES를 넘으면 output.json을 기다리지 않고
    즉시 kill + 원인이 명시된 error_trace를 반환한다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd, **kwargs)
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    over_limit = 5 * 1024 ** 3
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc), \
         patch("runtime.isolate._read_rss_bytes", return_value=over_limit), \
         patch.dict("os.environ", {"EVAL_RSS_LIMIT_BYTES": str(4 * 1024 ** 3)}):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
        )
    assert result.error_trace is not None
    assert "memory watchdog" in result.error_trace
    assert result.peak_rss_bytes == over_limit


def test_eval_isolated_kills_on_cpu_budget_exceeded() -> None:
    """peak CPU 시간이 예산을 넘으면 output.json을 기다리지 않고 즉시 kill +
    'cpu budget exceeded'가 명시된 error_trace를 반환한다 — 과거엔 커널이
    SIGKILL(rc=-9)로 흔적 없이 죽여 OOM과 구분이 안 됐다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd, **kwargs)
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    over_budget = 950.0
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc), \
         patch("runtime.isolate._read_cpu_seconds", return_value=over_budget):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=900,
        )
    assert result.error_trace is not None
    assert "cpu budget exceeded" in result.error_trace
    assert result.peak_cpu_sec == over_budget


def _kill_result(*, progress_lines: list[str] | None, reason: str = "cpu"):
    """워치독 kill을 재현한다. progress_lines가 있으면 runner가 남긴 것처럼 _progress.log를 미리 채운다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def _on_init(self, cmd) -> None:
            if progress_lines is not None:
                (Path(cmd[2]) / "_progress.log").write_text("\n".join(progress_lines) + "\n")

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    patches = [patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc)]
    if reason == "cpu":
        patches.append(patch("runtime.isolate._read_cpu_seconds", return_value=950.0))
    else:
        patches.append(patch("runtime.isolate._read_rss_bytes", return_value=5 * 1024 ** 3))
    with patches[0], patches[1]:
        return eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True, cpu_budget_sec=900,
        )


def test_kill_appends_the_last_progress_line_below_the_unchanged_message() -> None:
    """#421: 첫 줄은 기존 kill 메시지 그대로(startswith 분기 유지)이고 둘째 줄이 마지막 진행 위치다."""
    result = _kill_result(progress_lines=[
        "stage=eval_start cpu=1", "stage=preselect_done cpu=800", "stage=fold_done fold=1/3 cpu=1650",
    ])

    first, second = result.error_trace.split("\n")
    assert first == "cpu budget exceeded: 950s CPU used (limit 900s)"
    assert second == "[last_progress] stage=fold_done fold=1/3 cpu=1650"


def test_kill_before_the_evaluation_started_reports_none() -> None:
    result = _kill_result(progress_lines=None)

    assert result.error_trace.endswith("\n[last_progress] none")


def test_memory_watchdog_kill_also_reports_the_last_progress_line() -> None:
    result = _kill_result(progress_lines=["stage=preselect_done cpu=70"], reason="memory")

    assert result.error_trace.startswith("memory watchdog:")
    assert result.error_trace.endswith("\n[last_progress] stage=preselect_done cpu=70")


def _fold1_run(
    progress_lines: list[str] | None, cpu: float, *, collect_oof: bool = False, budget: float = 900.0, n_splits: int = 3,
):
    """워치독이 한 번 폴링하는 시점에 CPU가 cpu이고 runner가 progress_lines를 남긴 상태를 재현한다."""
    import json as _json

    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def _on_init(self, cmd) -> None:
            tmpdir = Path(cmd[2])
            if progress_lines is not None:
                (tmpdir / "_progress.log").write_text("\n".join(progress_lines) + "\n")
            (tmpdir / "output.json").write_text(_json.dumps({"cv_score": 0.9, "error_trace": None}))

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc), \
         patch("runtime.isolate._read_cpu_seconds", return_value=cpu):
        return eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=n_splits, seed=42, is_classification=True, collect_oof=collect_oof,
            cpu_budget_sec=budget,
        )


_IN_FOLD1 = ["stage=eval_start cpu=1", "stage=preselect_done cpu=100"]


def test_fold1_projection_kills_a_runaway_before_fold1_ends() -> None:
    """#444: fold-1 한 번이 예산 전체를 쓰는 폭주는 fold-1 종료 후 투영(ADR-056)에 닿지 못한다.
    투영 = 루프 시작 CPU + n_splits x (지금 - 루프 시작 CPU) = 100 + 3 x 400 = 1300 > 900 x 1.15."""
    result = _fold1_run(_IN_FOLD1, cpu=500.0)

    first, second = result.error_trace.split("\n")
    assert first == "cpu budget exceeded: projected 1300s CPU during fold 1 (limit 900s)"
    assert second == "[last_progress] stage=preselect_done cpu=100"
    assert result.peak_cpu_sec == 500.0


def test_fold1_projection_boundary_follows_the_harness_margin() -> None:
    from evaluator.harness import _CPU_PROJECTION_MARGIN

    at_limit = 100.0 + (900.0 * _CPU_PROJECTION_MARGIN - 100.0) / 3
    assert _fold1_run(_IN_FOLD1, cpu=at_limit - 1).error_trace is None
    assert "projected" in _fold1_run(_IN_FOLD1, cpu=at_limit + 1).error_trace


def test_fold1_projection_lets_a_run_within_budget_finish() -> None:
    result = _fold1_run(_IN_FOLD1, cpu=300.0)

    assert result.error_trace is None
    assert result.cv_score == 0.9


def test_fold1_projection_skips_collect_oof_evaluations() -> None:
    """merge-verify 등은 완전한 점수가 필요해 harness의 투영과 같이 제외한다 — 예산(900s) 안이면 끝까지 돈다."""
    assert _fold1_run(_IN_FOLD1, cpu=500.0, collect_oof=True).error_trace is None


@pytest.mark.parametrize(
    "progress_lines",
    [None, ["stage=eval_start cpu=1"], _IN_FOLD1 + ["stage=cv_done cpu=700"]],
)
def test_projection_only_applies_while_the_cv_loop_runs(progress_lines) -> None:
    """preselect 중(eval_start)이거나 CV가 끝난 뒤(cv_done), 진행 기록이 없을 때는 일반 예산(900s)만 적용된다."""
    assert _fold1_run(progress_lines, cpu=800.0).error_trace is None


# 예산 900s, preselect 한도 = 900 x 0.5 = 450. 후보 6개 중 1개가 100s에 끝남(평균 100s) -> 종료 투영 = 110 + max(지금 - 110, 100) + 4 x 100.
_PRESELECT = ["stage=eval_start cpu=1", "stage=preselect_start cand=0/6 cpu=10", "stage=preselect_fit cand=1/6 cpu=110"]


def test_preselect_projection_kills_a_candidate_list_that_would_exhaust_the_budget() -> None:
    """ADR-068: 후보 6개를 전부 재학습하는 preselect가 예산의 절반을 넘길 것으로 투영되면 CV 전에 끊는다. 투영 = 110 + 100 + 400 = 610 > 450."""
    result = _fold1_run(_PRESELECT, cpu=120.0)

    first, second = result.error_trace.split("\n")
    assert first == "cpu budget exceeded: projected 610s CPU during preselect candidate 2/6 (limit 900s)"
    assert second == "[last_progress] stage=preselect_fit cand=1/6 cpu=110"


def test_preselect_projection_uses_the_in_progress_candidate_when_it_costs_more_than_the_average() -> None:
    """평균(100s)보다 오래 걸리는 후보가 진행 중이면 그 소모를 쓴다: 110 + 390 + 400 = 900."""
    assert "projected 900s CPU during preselect candidate 2/6" in _fold1_run(_PRESELECT, cpu=500.0).error_trace


def test_preselect_projection_lets_a_cheap_candidate_list_finish() -> None:
    cheap = ["stage=eval_start cpu=1", "stage=preselect_start cand=0/6 cpu=10", "stage=preselect_fit cand=1/6 cpu=60"]
    assert _fold1_run(cheap, cpu=70.0).error_trace is None  # 60 + 50 + 4 x 50 = 310 <= 450


def test_preselect_projection_waits_for_the_first_candidate() -> None:
    """첫 후보가 끝나기 전에는 후보 비용 근거가 없어 투영하지 않는다 — 일반 예산(900s)만 적용된다."""
    started = ["stage=eval_start cpu=1", "stage=preselect_start cand=0/6 cpu=10"]
    assert _fold1_run(started, cpu=800.0).error_trace is None


def test_preselect_projection_stops_once_preselect_is_done() -> None:
    assert _fold1_run(_PRESELECT + ["stage=preselect_done cpu=600"], cpu=620.0).error_trace is None


def test_preselect_projection_skips_collect_oof_evaluations() -> None:
    assert _fold1_run(_PRESELECT, cpu=120.0, collect_oof=True).error_trace is None


_FIVE_FOLDS = dict(budget=3600.0, n_splits=5)  # 한도 = 3600 x 1.15 = 4140


def test_rolling_projection_kills_when_the_completed_folds_imply_an_overrun() -> None:
    """fold-1이 이후 fold보다 싸서 fold-1 투영을 통과해도, 완료 fold 평균으로 어림한 총비용이 한도를 넘으면 그 시점에 끊는다(#448).
    투영 = 마지막 fold 종료 CPU + 남은 fold 수 x max(진행 중 fold 소모, 완료 fold 평균) = 1801 + 3 x 900 = 4501."""
    lines = ["stage=preselect_done cpu=1", "stage=fold_done fold=1/5 cpu=901", "stage=fold_done fold=2/5 cpu=1801"]
    result = _fold1_run(lines, cpu=1850.0, **_FIVE_FOLDS)

    first, second = result.error_trace.split("\n")
    assert first == "cpu budget exceeded: projected 4501s CPU during fold 3 (limit 3600s)"
    assert second == "[last_progress] stage=fold_done fold=2/5 cpu=1801"


def test_rolling_projection_lets_a_run_within_the_margin_finish() -> None:
    lines = ["stage=preselect_done cpu=1", "stage=fold_done fold=1/5 cpu=701", "stage=fold_done fold=2/5 cpu=1401"]
    assert _fold1_run(lines, cpu=1450.0, **_FIVE_FOLDS).error_trace is None  # 1401 + 3 x 700 = 3501 <= 4140


def test_rolling_projection_kills_a_runaway_fold_after_a_cheap_fold_1() -> None:
    """fold-1은 257s인데 fold-2가 1300s째 안 끝나는 폭주(s5e8 feature_engineering) — 진행 중 fold의 소모가 평균보다 크면 그 값으로 어림한다."""
    lines = ["stage=preselect_done cpu=1", "stage=fold_done fold=1/5 cpu=257"]
    result = _fold1_run(lines, cpu=1300.0, **_FIVE_FOLDS)

    assert result.error_trace.startswith("cpu budget exceeded: projected 4429s CPU during fold 2 (limit 3600s)")


def test_rolling_projection_stops_after_the_last_fold() -> None:
    lines = ["stage=preselect_done cpu=1"] + [f"stage=fold_done fold={k}/5 cpu={k * 700}" for k in range(1, 6)]
    assert _fold1_run(lines, cpu=3550.0, **_FIVE_FOLDS).error_trace is None  # holdout 등 CV 이후 단계는 투영하지 않는다


def test_rolling_projection_skips_collect_oof_evaluations() -> None:
    lines = ["stage=preselect_done cpu=1", "stage=fold_done fold=1/5 cpu=901", "stage=fold_done fold=2/5 cpu=1801"]
    assert _fold1_run(lines, cpu=1850.0, collect_oof=True, **_FIVE_FOLDS).error_trace is None


def test_eval_isolated_cpu_budget_sec_overrides_env_default() -> None:
    """호출자가 넘긴 cpu_budget_sec이 EVAL_CPU_BUDGET_SECS 기본값보다 우선한다 —
    cycle/run.py가 attempt 단위로 남은 예산을 재시도마다 다르게 넘기기 위함."""
    import polars as pl

    from runtime.isolate import eval_isolated

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakePopen), \
         patch("runtime.isolate._read_cpu_seconds", return_value=50.0), \
         patch.dict("os.environ", {"EVAL_CPU_BUDGET_SECS": "900"}):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=30,
        )
    assert result.error_trace is not None
    assert "cpu budget exceeded" in result.error_trace
    assert "limit 30s" in result.error_trace


def _input_json_seen_by_runner(**kwargs) -> dict:
    import json as _json

    import polars as pl

    from runtime.isolate import eval_isolated

    seen: dict = {}

    class _FakeProc(_FakePopen):
        def _on_init(self, cmd) -> None:
            tmpdir = Path(cmd[2])
            seen.update(_json.loads((tmpdir / "input.json").read_text()))
            (tmpdir / "output.json").write_text(_json.dumps({"cv_score": 0.9, "error_trace": None}))

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc):
        eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True, **kwargs,
        )
    return seen


def test_eval_isolated_passes_the_cpu_budget_to_the_runner() -> None:
    """#361: runner가 fold-1 CPU 투영에 쓸 예산이 input.json으로 넘어간다 — watchdog이 집행하는 값과 같아야 한다."""
    assert _input_json_seen_by_runner(cpu_budget_sec=1234)["cpu_budget_sec"] == 1234


def test_eval_isolated_passes_the_env_default_budget_when_none_is_given() -> None:
    with patch.dict("os.environ", {"EVAL_CPU_BUDGET_SECS": "777"}):
        assert _input_json_seen_by_runner()["cpu_budget_sec"] == 777.0


def test_eval_isolated_passes_known_fold1_scores_to_the_runner() -> None:
    """#376: fold-1 행동 지문 캐시가 runner의 input.json으로 넘어가야 한다."""
    cache = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert _input_json_seen_by_runner(known_fold1_scores=cache)["known_fold1_scores"] == cache


def test_eval_isolated_defaults_known_fold1_scores_to_none() -> None:
    assert _input_json_seen_by_runner()["known_fold1_scores"] is None


def _exits_after(n_polls: int):
    """n_polls번 폴링된 뒤 스스로 종료하는 가짜 subprocess."""
    class _Proc:
        pid = 4321
        returncode = 0

        def __init__(self, *a, **kw) -> None:
            self.calls = 0
            self.killed = False

        def wait(self, timeout=None):
            self.calls += 1
            if timeout is not None and self.calls <= n_polls:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0

        def kill(self) -> None:
            self.killed = True

    return _Proc


def test_wall_timeout_defaults_to_at_least_cpu_budget() -> None:
    """벽시계 상한이 CPU 예산보다 낮으면 예산을 선점해 무력화한다(#207) — 호출자가
    timeout_sec을 안 주면 CPU 예산이 항상 먼저 걸리도록 벽시계를 그만큼 늘린다."""
    import polars as pl

    from runtime.isolate import DEFAULT_TIMEOUT, eval_isolated

    assert DEFAULT_TIMEOUT < 3000  # 예산이 기본 벽시계보다 커야 의미 있는 케이스

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    clock = iter([0.0] + [float(DEFAULT_TIMEOUT) + 1.0] * 50)
    with patch("runtime.isolate.subprocess.Popen", side_effect=_exits_after(2)), \
         patch("runtime.isolate._read_cpu_seconds", return_value=10.0), \
         patch("runtime.isolate._read_rss_bytes", return_value=1000), \
         patch("runtime.isolate.time.monotonic", side_effect=lambda: next(clock)):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=3000,
        )
    assert result.error_trace is not None
    assert "timeout" not in result.error_trace


def test_explicit_timeout_sec_is_respected() -> None:
    """호출자가 timeout_sec을 명시하면 CPU 예산과 무관하게 그 값이 벽시계 상한이다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    clock = iter([0.0] + [60.0] * 50)
    with patch("runtime.isolate.subprocess.Popen", side_effect=_exits_after(50)), \
         patch("runtime.isolate._read_cpu_seconds", return_value=10.0), \
         patch("runtime.isolate._read_rss_bytes", return_value=1000), \
         patch("runtime.isolate.time.monotonic", side_effect=lambda: next(clock)):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=3000, timeout_sec=30,
        )
    assert result.error_trace is not None
    assert "timeout after 30s" in result.error_trace


def test_err_result_defaults_gain_relative_to_none() -> None:
    from runtime.isolate import _err
    result = _err("some failure")
    assert result.gain_vs_best_relative is None
    assert result.peak_rss_bytes is None
    assert result.peak_cpu_sec is None


def test_err_result_carries_peak_rss_bytes() -> None:
    from runtime.isolate import _err
    result = _err("memory watchdog: ...", peak_rss_bytes=123)
    assert result.peak_rss_bytes == 123


def test_err_result_carries_peak_cpu_sec() -> None:
    from runtime.isolate import _err
    result = _err("cpu budget exceeded: ...", peak_cpu_sec=456.0)
    assert result.peak_cpu_sec == 456.0
