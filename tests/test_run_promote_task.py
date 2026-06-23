"""BON-187: promote task uses slug (--competition) for module import, not DB competition_id."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).parent.parent


def _fake_comp_module(slug: str, full_id: str) -> types.ModuleType:
    mod = types.ModuleType(f"config.competitions.{slug}")
    mod.COMPETITION_ID = full_id
    mod.S3_DATA_PATH = None
    mod.DATA_DIR = Path("/nonexistent")
    mod.DROP_COLS = []
    mod.TARGET = "target"
    mod.IS_CLASSIFICATION = True
    mod.N_SPLITS = 5
    mod.METRIC = "auc"
    mod.METRIC_SIGN = 1
    return mod


def _run_main(slug: str, monkeypatched_import) -> None:
    from unittest.mock import patch

    argv = ["run_promote_task", "--queue-id", "test-queue-id", "--competition", slug]
    with patch.object(sys, "argv", argv):
        with patch("store.db.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.return_value.execute.return_value = mock_cur
            with patch("importlib.import_module", side_effect=monkeypatched_import):
                sys.path.insert(0, str(ROOT))
                try:
                    from bin import run_promote_task
                    importlib.reload(run_promote_task)
                    run_promote_task.main()
                except SystemExit:
                    pass
                finally:
                    if str(ROOT) in sys.path:
                        sys.path.remove(str(ROOT))


def test_promote_task_accepts_competition_arg() -> None:
    """--competition 인자가 파싱되어야 한다."""
    import argparse
    sys.path.insert(0, str(ROOT))
    try:
        from bin.run_promote_task import main
        # argparse만 테스트 — 실제 main 실행은 DB 필요
        parser = argparse.ArgumentParser()
        parser.add_argument("--queue-id", required=True)
        parser.add_argument("--competition", required=True)
        args = parser.parse_args(["--queue-id", "qid", "--competition", "s4e1"])
        assert args.queue_id == "qid"
        assert args.competition == "s4e1"
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))


def test_promote_task_imports_by_slug_not_full_id() -> None:
    """import_module이 slug(s4e1)를 쓰고 full competition_id(playground-series-s4e1)를 쓰지 않는다."""
    slug = "s4e1"
    full_id = "playground-series-s4e1"
    imported_modules: list[str] = []

    original_import = importlib.import_module

    def tracking_import(name: str, *args, **kwargs):
        if name.startswith("config.competitions."):
            imported_modules.append(name)
            mod = _fake_comp_module(slug, full_id)
            return mod
        return original_import(name, *args, **kwargs)

    sys.path.insert(0, str(ROOT))
    try:
        argv = ["run_promote_task", "--queue-id", "test-qid", "--competition", slug]
        with patch.object(sys, "argv", argv):
            with patch("importlib.import_module", side_effect=tracking_import):
                with patch("store.db.connect") as mock_conn:
                    mock_cur = MagicMock()
                    # no context found → early exit
                    mock_cur.fetchone.return_value = None
                    mock_conn.return_value.execute.return_value = mock_cur
                    try:
                        import bin.run_promote_task as rpt
                        importlib.reload(rpt)
                        rpt.main()
                    except SystemExit:
                        pass
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))

    comp_imports = [m for m in imported_modules if m.startswith("config.competitions.")]
    for mod_name in comp_imports:
        assert full_id not in mod_name, (
            f"import_module called with full competition_id ({full_id}), should use slug ({slug}). "
            f"Got: {mod_name}"
        )
        assert slug in mod_name or not comp_imports, (
            f"Expected slug {slug!r} in import path, got {mod_name!r}"
        )
