"""Shared pytest fixtures for the ServerNew API tests.

The application is configured via the ``DATABASE_URL`` environment variable at
import time. We point it at a fresh SQLite database for each test so that the
tests are fully isolated and don't require a running PostgreSQL server.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import BigInteger, event
from sqlalchemy.ext.compiler import compiles


# SQLite only auto-increments ``INTEGER PRIMARY KEY``; compile ``BigInteger``
# to ``INTEGER`` for the SQLite dialect so the schema works in tests.
@compiles(BigInteger, "sqlite")
def _bigint_as_integer_in_sqlite(_element, _compiler, **_kw):
    return "INTEGER"


# Make ``main`` importable as a top-level module.
SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """Return a TestClient bound to a fresh SQLite-backed app instance."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Force a fresh import so ``create_engine`` picks up the test DATABASE_URL.
    sys.modules.pop("main", None)
    main = importlib.import_module("main")

    # SQLite has no built-in ``char_length`` function (the schema uses it for a
    # CHECK constraint). Register a Python implementation per-connection.
    @event.listens_for(main.engine, "connect")
    def _register_sqlite_funcs(dbapi_conn, _):  # pragma: no cover - trivial glue
        try:
            dbapi_conn.create_function("char_length", 1, lambda s: len(s) if s is not None else 0)
        except AttributeError:
            pass

    # Create the schema (the app normally does this in its startup hook).
    main.Base.metadata.create_all(bind=main.engine)

    with TestClient(main.app) as client:
        client.app_module = main  # type: ignore[attr-defined]
        yield client

    main.engine.dispose()
    sys.modules.pop("main", None)


def _create_user(client: TestClient, user_name="alice", password="pw", email="a@b.c"):
    """Helper: create a user, returning the activation code from the DB."""
    resp = client.post(
        "/createuser",
        data={"userName": user_name, "password": password, "email": email},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": True}

    main = client.app_module  # type: ignore[attr-defined]
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, user_name)
        assert user is not None
        return user.activation_code


def _activate_user(client: TestClient, user_name="alice", password="pw", code=None, gpu=None):
    if code is None:
        main = client.app_module  # type: ignore[attr-defined]
        with main.Session(main.engine) as session:
            user = main._get_user_by_name(session, user_name)
            code = user.activation_code
    data = {"userName": user_name, "password": password, "activationCode": code}
    if gpu is not None:
        data["gpu"] = gpu
    return client.post("/activateuser", data=data)


def _create_active_user(client: TestClient, user_name="alice", password="pw", email="a@b.c"):
    """Helper: create + activate a user, returning the user's database row."""
    code = _create_user(client, user_name, password, email)
    resp = _activate_user(client, user_name, password, code=code)
    assert resp.status_code == 200
    assert resp.json() == {"result": True}
    main = client.app_module  # type: ignore[attr-defined]
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, user_name)
        assert user is not None
        return user


def _upload_simulation(
    client: TestClient,
    user_name: str,
    password: str,
    sim_name: str = "sim",
    description: str = "desc",
    width: int = 16,
    height: int = 8,
    particles: int = 100,
    version: str = "1.0",
    content: bytes = b"\x00\x01\x02binary-payload",
    settings: str = "{}",
    statistics: str = "stats",
    sim_type: int = 0,
    workspace: int = 0,
    symbol_map: str = "",
):
    files = {
        "userName": (None, user_name),
        "password": (None, password),
        "simName": (None, sim_name),
        "simDesc": (None, description),
        "width": (None, str(width)),
        "height": (None, str(height)),
        "particles": (None, str(particles)),
        "version": (None, version),
        "content": ("", content, "application/octet-stream"),
        "settings": (None, settings),
        "symbolMap": (None, symbol_map),
        "type": (None, str(sim_type)),
        "workspace": (None, str(workspace)),
        "statistics": (None, statistics),
    }
    return client.post("/uploadsimulation", files=files)


@pytest.fixture()
def helpers():
    """Expose helper functions to tests as a small namespace."""

    class Helpers:
        create_user = staticmethod(_create_user)
        activate_user = staticmethod(_activate_user)
        create_active_user = staticmethod(_create_active_user)
        upload_simulation = staticmethod(_upload_simulation)

    return Helpers
