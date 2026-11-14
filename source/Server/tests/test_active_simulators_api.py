"""Tests for the active simulator count fed by ``/getversionedsimulationlist`` and ``/reportactivesimulator`` and its hourly history."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import secrets
import sys
from datetime import datetime, timezone

from fastapi.testclient import TestClient


def _client_from(app_client, host):
    return TestClient(app_client.app_module.app, client=(host, 50000))


def _fetch_list(client, **data):
    resp = client.post("/getversionedsimulationlist", data=data)
    assert resp.status_code == 200


def _report(client):
    resp = client.post("/reportactivesimulator")
    assert resp.status_code == 200
    assert resp.json() == {"result": True}


def _simulator_counts(client):
    resp = client.get("/getactivesimulatorcount")
    assert resp.status_code == 200
    return resp.json()


def _active_simulators(client):
    return _simulator_counts(client)["activeSimulators"]


def _fake_clock(main, monkeypatch, uptime):
    now = [1_000_000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(main, "_recording_start_time", now[0] - uptime)
    return now


def test_count_is_zero_without_list_fetches(app_client):
    assert _active_simulators(app_client) == 0


def test_list_fetches_are_counted_per_address(app_client):
    first = _client_from(app_client, "10.0.0.1")
    second = _client_from(app_client, "10.0.0.2")

    _fetch_list(first)
    _fetch_list(first)
    _fetch_list(second)

    assert _active_simulators(app_client) == 2


def test_opted_in_clients_are_counted(app_client):
    _fetch_list(_client_from(app_client, "10.0.0.1"), countAsActive="true")

    assert _active_simulators(app_client) == 1


def test_opted_out_clients_are_not_counted(app_client):
    _fetch_list(_client_from(app_client, "10.0.0.1"), countAsActive="false")

    assert _active_simulators(app_client) == 0


def test_reports_are_counted_per_address(app_client):
    first = _client_from(app_client, "10.0.0.1")
    second = _client_from(app_client, "10.0.0.2")

    _report(first)
    _report(first)
    _report(second)

    assert _active_simulators(app_client) == 2


def test_report_and_list_fetch_from_one_address_count_once(app_client):
    client = _client_from(app_client, "10.0.0.1")

    _report(client)
    _fetch_list(client)

    assert _active_simulators(app_client) == 1


def test_inactive_simulators_expire(app_client, monkeypatch):
    main = app_client.app_module
    now = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS / 2
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS / 2 + 1

    assert _active_simulators(app_client) == 1


def test_last_day_count_is_withheld_during_warm_up(app_client, monkeypatch):
    main = app_client.app_module
    _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS - 1)

    _fetch_list(_client_from(app_client, "10.0.0.1"))

    assert _simulator_counts(app_client) == {"activeSimulators": 1}


def test_last_day_count_includes_simulators_that_stopped(app_client, monkeypatch):
    main = app_client.app_module
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += 2 * 60 * 60
    _report(_client_from(app_client, "10.0.0.2"))

    assert _simulator_counts(app_client) == {"activeSimulators": 1, "lastDaySimulators": 2}


def test_last_day_simulators_expire(app_client, monkeypatch):
    main = app_client.app_module
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += main.LAST_DAY_SECONDS / 2
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    now[0] += main.LAST_DAY_SECONDS / 2 + 1

    assert _simulator_counts(app_client)["lastDaySimulators"] == 1


def test_ipv6_addresses_of_one_prefix_count_once(app_client):
    _fetch_list(_client_from(app_client, "2001:db8:1:2::1"))
    _fetch_list(_client_from(app_client, "2001:db8:1:2:abcd:ef01:2345:6789"))
    _fetch_list(_client_from(app_client, "2001:db8:1:3::1"))

    assert _active_simulators(app_client) == 2


def test_ipv4_mapped_addresses_count_as_ipv4(app_client):
    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _fetch_list(_client_from(app_client, "::ffff:10.0.0.1"))

    assert _active_simulators(app_client) == 1


def test_addresses_are_not_stored_in_plain_text(app_client):
    _fetch_list(_client_from(app_client, "10.0.0.1"))

    assert all("10.0.0.1" not in key for key in app_client.app_module._active_simulators)


def test_oldest_simulators_beyond_limit_are_dropped(app_client, monkeypatch):
    main = app_client.app_module
    monkeypatch.setattr(main, "MAX_ACTIVE_SIMULATORS", 2)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _fetch_list(_client_from(app_client, "10.0.0.3"))

    remembered = set(main._active_simulators)
    assert remembered == {hmac.new(main._active_simulator_key, host.encode("utf-8"), hashlib.sha256).hexdigest() for host in ("10.0.0.1", "10.0.0.3")}


def _history(client):
    resp = client.get("/getactivesimulatorhistory")
    assert resp.status_code == 200
    return resp.json()


def _utc(hour, minute=0):
    return datetime(2026, 10, 18, hour, minute, tzinfo=timezone.utc)


def _end_warm_up(main, monkeypatch):
    monkeypatch.setattr(main, "_recording_start_time", main.time.monotonic() - main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS)


def test_history_is_empty_without_snapshots(app_client):
    assert _history(app_client) == []


def test_history_is_readable_from_other_origins(app_client):
    resp = app_client.get("/getactivesimulatorhistory", headers={"Origin": "null"})

    assert resp.headers["access-control-allow-origin"] == "*"


def test_snapshots_store_the_active_simulator_count(app_client, monkeypatch):
    main = app_client.app_module
    _end_warm_up(main, monkeypatch)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    main._store_active_simulator_count(_utc(14))
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    main._store_active_simulator_count(_utc(15))

    assert _history(app_client) == [
        {"timestamp": "2026-10-18T14:00:00+00:00", "activeSimulators": 1},
        {"timestamp": "2026-10-18T15:00:00+00:00", "activeSimulators": 2},
    ]


def test_snapshots_store_the_last_day_count_after_its_warm_up(app_client, monkeypatch):
    main = app_client.app_module
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS + 1
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    main._store_active_simulator_count(_utc(14))

    assert _history(app_client) == [
        {"timestamp": "2026-10-18T14:00:00+00:00", "activeSimulators": 1, "lastDaySimulators": 2},
    ]


def test_schema_update_keeps_existing_snapshots(app_client):
    from sqlalchemy import text

    main = app_client.app_module
    with main.engine.begin() as conn:
        conn.execute(text("ALTER TABLE active_simulator_counts DROP COLUMN last_day_count"))
        conn.execute(text("INSERT INTO active_simulator_counts (timestamp, count) VALUES ('2026-10-18 14:00:00', 7)"))

    main._ensure_active_simulator_counts_schema()

    assert _history(app_client) == [{"timestamp": "2026-10-18T14:00:00+00:00", "activeSimulators": 7}]


def test_history_is_sorted_by_time(app_client, monkeypatch):
    main = app_client.app_module
    _end_warm_up(main, monkeypatch)

    main._store_active_simulator_count(_utc(15))
    main._store_active_simulator_count(_utc(14))

    assert [entry["timestamp"] for entry in _history(app_client)] == [
        "2026-10-18T14:00:00+00:00",
        "2026-10-18T15:00:00+00:00",
    ]


def test_snapshots_are_skipped_during_warm_up(app_client):
    app_client.app_module._store_active_simulator_count(_utc(14))

    assert _history(app_client) == []


class _FakeStopEvent:
    def __init__(self, clock, stop_time):
        self._clock = clock
        self._stop_time = stop_time

    def wait(self, seconds):
        self._clock[0] += seconds
        return self._clock[0] > self._stop_time


def test_snapshots_are_taken_at_full_hours(app_client, monkeypatch):
    main = app_client.app_module
    _end_warm_up(main, monkeypatch)
    clock = [_utc(13, 30).timestamp()]

    with monkeypatch.context() as patch:
        patch.setattr(main.time, "time", lambda: clock[0])
        patch.setattr(main, "_active_simulator_history_stop", _FakeStopEvent(clock, _utc(15, 30).timestamp()))
        main._store_active_simulator_counts_hourly()

    assert [entry["timestamp"] for entry in _history(app_client)] == [
        "2026-10-18T14:00:00+00:00",
        "2026-10-18T15:00:00+00:00",
    ]


def _use_runtime_directory(monkeypatch, tmp_path):
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    monkeypatch.setenv("RUNTIME_DIRECTORY", str(runtime_directory))
    return runtime_directory


def _restart(main, monkeypatch, now, downtime):
    wall_clock = [1_800_000_000.0]
    with monkeypatch.context() as patch:
        patch.setattr(main.time, "time", lambda: wall_clock[0])
        main.hand_over_active_simulators()
        wall_clock[0] += downtime
        now[0] += downtime
        main._active_simulators.clear()
        main._active_simulator_key = secrets.token_bytes(32)
        main._recording_start_time = now[0]
        main.take_over_active_simulators()


def test_restart_keeps_the_counts(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += 2 * 60 * 60
    _report(_client_from(app_client, "10.0.0.2"))
    _restart(main, monkeypatch, now, 5)

    assert _simulator_counts(app_client) == {"activeSimulators": 1, "lastDaySimulators": 2}


def test_restarted_server_takes_over_the_counts(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "_recording_start_time", main.time.monotonic() - main.LAST_DAY_SECONDS)
    _fetch_list(_client_from(app_client, "10.0.0.1"))

    with TestClient(main.app):
        pass
    sys.modules.pop("main")
    restarted = importlib.import_module("main")
    try:
        with TestClient(restarted.app) as client:
            assert _simulator_counts(client) == {"activeSimulators": 1, "lastDaySimulators": 1}
    finally:
        restarted.engine.dispose()


def test_restart_recognizes_known_callers(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _restart(main, monkeypatch, now, 5)
    _fetch_list(_client_from(app_client, "10.0.0.1"))

    assert _simulator_counts(app_client) == {"activeSimulators": 1, "lastDaySimulators": 1}


def test_downtime_adds_to_the_age_of_callers(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS - 10
    _restart(main, monkeypatch, now, 20)

    assert _simulator_counts(app_client) == {"activeSimulators": 0, "lastDaySimulators": 1}


def test_snapshots_continue_after_a_restart(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _restart(main, monkeypatch, now, 5)
    main._store_active_simulator_count(_utc(14))

    assert _history(app_client) == [{"timestamp": "2026-10-18T14:00:00+00:00", "activeSimulators": 1, "lastDaySimulators": 1}]


def test_handover_is_deleted_after_the_take_over(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    runtime_directory = _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _restart(main, monkeypatch, now, 5)

    assert list(runtime_directory.iterdir()) == []


def test_restart_after_a_long_downtime_starts_anew(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    runtime_directory = _use_runtime_directory(monkeypatch, tmp_path)
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _restart(main, monkeypatch, now, main.ACTIVE_SIMULATOR_HANDOVER_MAX_DOWNTIME_SECONDS + 1)

    assert _simulator_counts(app_client) == {"activeSimulators": 0}
    assert list(runtime_directory.iterdir()) == []


def test_handover_of_another_version_is_discarded(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    _use_runtime_directory(monkeypatch, tmp_path)
    _fetch_list(_client_from(app_client, "10.0.0.1"))

    main.hand_over_active_simulators()
    monkeypatch.setattr(main, "ACTIVE_SIMULATOR_HANDOVER_VERSION", main.ACTIVE_SIMULATOR_HANDOVER_VERSION + 1)
    main._active_simulators.clear()
    main.take_over_active_simulators()

    assert _simulator_counts(app_client) == {"activeSimulators": 0}


def test_restart_without_runtime_directory_starts_anew(app_client, monkeypatch):
    main = app_client.app_module
    now = _fake_clock(main, monkeypatch, main.LAST_DAY_SECONDS)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _restart(main, monkeypatch, now, 5)

    assert _simulator_counts(app_client) == {"activeSimulators": 0}


def test_unreadable_handover_is_discarded(app_client, monkeypatch, tmp_path):
    main = app_client.app_module
    runtime_directory = _use_runtime_directory(monkeypatch, tmp_path)
    (runtime_directory / main.ACTIVE_SIMULATOR_HANDOVER_FILE_NAME).write_text("{", encoding="utf-8")

    main.take_over_active_simulators()

    assert _simulator_counts(app_client) == {"activeSimulators": 0}
    assert list(runtime_directory.iterdir()) == []
