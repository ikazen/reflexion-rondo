from bin.run_daemon import _final_status


def test_all_success():
    assert _final_status(cycles_done=3, skipped=0, failed_cycles=0) == ("done", None)


def test_partial_failure_still_done():
    assert _final_status(cycles_done=2, skipped=0, failed_cycles=1) == ("done", None)


def test_all_failed():
    status, err = _final_status(cycles_done=0, skipped=0, failed_cycles=3)
    assert status == "failed"
    assert err is not None


def test_all_skipped():
    status, err = _final_status(cycles_done=0, skipped=2, failed_cycles=0)
    assert status == "failed"
    assert err is not None


def test_empty_batch():
    assert _final_status(cycles_done=0, skipped=0, failed_cycles=0) == ("done", None)


def test_mixed_failed_and_skipped():
    status, err = _final_status(cycles_done=0, skipped=1, failed_cycles=2)
    assert status == "failed"
    assert "2 failed" in err
    assert "1 skipped" in err
