"""Tests for the active simulator count fed by ``/getversionedsimulationlist`` and its hourly history."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient


def _client_from(app_client, host):
    return TestClient(app_client.app_module.app, client=(host, 50000))


def _fetch_list(client, **data):
    resp = client.post("/getversionedsimulationlist", data=data)
    assert resp.status_code == 200


def _active_simulators(client):
    resp = client.get("/getactivesimulatorcount")
    assert resp.status_code == 200
    return resp.json()["activeSimulators"]


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


def test_inactive_simulators_expire(app_client, monkeypatch):
    main = app_client.app_module
    now = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS / 2
    _fetch_list(_client_from(app_client, "10.0.0.2"))
    now[0] += main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS / 2 + 1

    assert _active_simulators(app_client) == 1


def test_addresses_are_not_stored_in_plain_text(app_client):
    _fetch_list(_client_from(app_client, "10.0.0.1"))

    assert all("10.0.0.1" not in key for key in app_client.app_module._active_simulators)


def test_new_simulators_beyond_limit_are_ignored(app_client, monkeypatch):
    monkeypatch.setattr(app_client.app_module, "MAX_ACTIVE_SIMULATORS", 1)

    _fetch_list(_client_from(app_client, "10.0.0.1"))
    _fetch_list(_client_from(app_client, "10.0.0.2"))

    assert _active_simulators(app_client) == 1


def _history(client):
    resp = client.get("/getactivesimulatorhistory")
    assert resp.status_code == 200
    return resp.json()


def _utc(hour, minute=0):
    return datetime(2026, 10, 18, hour, minute, tzinfo=timezone.utc)


def _end_warm_up(main, monkeypatch):
    monkeypatch.setattr(main, "_server_start_time", main.time.monotonic() - main.ACTIVE_SIMULATOR_TIMEOUT_SECONDS)


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
