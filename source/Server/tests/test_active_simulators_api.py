"""Tests for the active simulator count fed by ``/getversionedsimulationlist``."""

from __future__ import annotations

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
