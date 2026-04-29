"""End-to-end tests for the FastAPI user-management endpoints in main.py.

Each endpoint of NetworkService (createUser, activateUser, login, logout,
refreshLogin, deleteUser, resetPassword, setNewPassword) is exercised against a
fresh SQLite-backed app instance.
"""

from __future__ import annotations


# --- /health (smoke test) -----------------------------------------------------
def test_health_endpoint(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- /createuser --------------------------------------------------------------
def test_create_user_succeeds_and_persists_pending_user(app_client):
    resp = app_client.post(
        "/createuser",
        data={"userName": "alice", "password": "pw", "email": "a@b.c"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert user is not None
        assert user.flags == 0
        # Activation code is set (user not yet activated).
        assert user.activation_code
        assert len(user.activation_code) == 6
        # Password hash is sha256(pw + salt).
        assert user.pw_hash == main._hash_password("pw", user.salt)
        assert user.email_hash == main._hash_email("a@b.c")


def test_create_user_rejects_name_with_spaces(app_client):
    resp = app_client.post(
        "/createuser",
        data={"userName": "bad name", "password": "pw", "email": "x@y.z"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": False}


def test_create_user_rejects_empty_name(app_client):
    resp = app_client.post(
        "/createuser",
        data={"userName": "", "password": "pw", "email": "x@y.z"},
    )
    # Either FastAPI's form validation rejects it (422) or our handler does
    # (200 with result=False). Both outcomes mean the user was not created.
    if resp.status_code == 200:
        assert resp.json() == {"result": False}
    else:
        assert resp.status_code == 422

    main = app_client.app_module
    with main.Session(main.engine) as session:
        assert main._get_user_by_name(session, "") is None


def test_create_user_replaces_pending_record_with_same_name(app_client, helpers):
    code1 = helpers.create_user(app_client, "alice", "pw1", "a@b.c")
    # Same name, while still pending activation: must succeed and replace the row.
    code2 = helpers.create_user(app_client, "alice", "pw2", "a@b.c")
    assert code1 != code2

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        # New password must be in effect; old one must not work anymore.
        assert main._check_password_and_activation_code(user, "pw2", code2)
        assert not main._check_password_and_activation_code(user, "pw1", code1)


def test_create_user_rejects_existing_activated_name(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/createuser",
        data={"userName": "alice", "password": "other", "email": "z@z.z"},
    )
    assert resp.json() == {"result": False}


# --- /activateuser ------------------------------------------------------------
def test_activate_user_clears_activation_code(app_client, helpers):
    code = helpers.create_user(app_client, "alice", "pw", "a@b.c")

    resp = helpers.activate_user(app_client, "alice", "pw", code=code)
    assert resp.status_code == 200
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert main._is_activated(user)


def test_activate_user_with_wrong_code_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    resp = helpers.activate_user(app_client, "alice", "pw", code="000000")
    assert resp.json() == {"result": False}


def test_activate_user_with_wrong_password_fails(app_client, helpers):
    code = helpers.create_user(app_client, "alice", "pw", "a@b.c")
    resp = helpers.activate_user(app_client, "alice", "wrong", code=code)
    assert resp.json() == {"result": False}


def test_activate_user_unknown_user_fails(app_client, helpers):
    resp = helpers.activate_user(app_client, "ghost", "pw", code="abcdef")
    assert resp.json() == {"result": False}


def test_activate_user_accepts_optional_gpu(app_client, helpers):
    code = helpers.create_user(app_client, "alice", "pw", "a@b.c")
    resp = helpers.activate_user(app_client, "alice", "pw", code=code, gpu="RTX 4090")
    assert resp.json() == {"result": True}


# --- /login -------------------------------------------------------------------
def test_login_success_sets_flags_and_gpu(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/login",
        data={"userName": "alice", "password": "pw", "gpu": "RTX 4090"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": True, "errorCode": 1}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert user.flags == 1
        assert user.gpu == "RTX 4090"


def test_login_unknown_user_returns_error_code_1(app_client):
    resp = app_client.post("/login", data={"userName": "ghost", "password": "pw"})
    assert resp.json() == {"result": False, "errorCode": 1}


def test_login_not_activated_returns_error_code_0(app_client, helpers):
    # Create but do NOT activate.
    helpers.create_user(app_client, "alice", "pw", "a@b.c")

    resp = app_client.post("/login", data={"userName": "alice", "password": "pw"})
    assert resp.json() == {"result": False, "errorCode": 0}


def test_login_wrong_password_returns_error_code_1(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post("/login", data={"userName": "alice", "password": "wrong"})
    assert resp.json() == {"result": False, "errorCode": 1}


# --- /logout ------------------------------------------------------------------
def test_logout_clears_flags(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")
    app_client.post("/login", data={"userName": "alice", "password": "pw"})

    resp = app_client.post("/logout", data={"userName": "alice", "password": "pw"})
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert user.flags == 0


def test_logout_with_wrong_password_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post("/logout", data={"userName": "alice", "password": "wrong"})
    assert resp.json() == {"result": False}


# --- /refreshlogin ------------------------------------------------------------
def test_refresh_login_increments_time_spent(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    for expected in (1, 2, 3):
        resp = app_client.post(
            "/refreshlogin", data={"userName": "alice", "password": "pw"}
        )
        assert resp.json() == {"result": True}

        main = app_client.app_module
        with main.Session(main.engine) as session:
            user = main._get_user_by_name(session, "alice")
            assert user.flags == 1
            assert user.time_spent == expected


def test_refresh_login_with_wrong_password_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/refreshlogin", data={"userName": "alice", "password": "wrong"}
    )
    assert resp.json() == {"result": False}


# --- /deleteuser --------------------------------------------------------------
def test_delete_user_removes_row(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/deleteuser", data={"userName": "alice", "password": "pw"}
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        assert main._get_user_by_name(session, "alice") is None


def test_delete_user_cascades_to_userlikes_and_simulations(app_client, helpers):
    """deleteUser must also remove the user's simulations and all userlikes
    that reference either the user or any of those simulations (foreign-key
    dependencies on the ``users`` table)."""
    main = app_client.app_module
    from sqlalchemy import select as _select

    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")

    # Alice owns one simulation, Bob owns one.
    resp = helpers.upload_simulation(app_client, "alice", "pw", sim_name="alice-sim")
    assert resp.json()["result"] is True
    alice_sim_id = int(resp.json()["simId"])
    resp = helpers.upload_simulation(app_client, "bob", "pw2", sim_name="bob-sim")
    assert resp.json()["result"] is True
    bob_sim_id = int(resp.json()["simId"])

    # Alice likes Bob's sim; Bob likes Alice's sim.
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "alice", "password": "pw", "simId": str(bob_sim_id), "likeType": "1"},
    ).json() == {"result": True}
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(alice_sim_id), "likeType": "2"},
    ).json() == {"result": True}

    with main.Session(main.engine) as session:
        assert session.execute(_select(main.UserLike)).all()  # sanity: 2 likes exist
        assert session.execute(_select(main.Simulation)).all()  # 2 sims exist

    # Delete Alice.
    assert app_client.post(
        "/deleteuser", data={"userName": "alice", "password": "pw"}
    ).json() == {"result": True}

    with main.Session(main.engine) as session:
        # Alice's user row gone.
        assert main._get_user_by_name(session, "alice") is None
        # Alice's simulation gone.
        assert session.get(main.Simulation, alice_sim_id) is None
        # Bob's simulation still there.
        assert session.get(main.Simulation, bob_sim_id) is not None

        # All remaining userlike rows must belong to Bob (the only surviving
        # user) AND must reference Bob's surviving simulation. The two original
        # likes (Alice→Bob's sim, Bob→Alice's sim) both involve Alice via FK
        # and must be gone.
        bob_user_id = main._get_user_by_name(session, "bob").id
        remaining = session.execute(_select(main.UserLike)).scalars().all()
        for like in remaining:
            assert like.user_id == bob_user_id
            assert like.simulation_id == bob_sim_id


def test_delete_user_wrong_password_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/deleteuser", data={"userName": "alice", "password": "wrong"}
    )
    assert resp.json() == {"result": False}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        assert main._get_user_by_name(session, "alice") is not None


def test_delete_user_unknown_user_fails(app_client):
    resp = app_client.post(
        "/deleteuser", data={"userName": "ghost", "password": "pw"}
    )
    assert resp.json() == {"result": False}


# --- /resetpw -----------------------------------------------------------------
def test_reset_password_sets_new_activation_code(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/resetpw", data={"userName": "alice", "email": "a@b.c"}
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert user.activation_code
        assert len(user.activation_code) == 6


def test_reset_password_wrong_email_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    resp = app_client.post(
        "/resetpw", data={"userName": "alice", "email": "wrong@x.y"}
    )
    assert resp.json() == {"result": False}


def test_reset_password_unknown_user_fails(app_client):
    resp = app_client.post(
        "/resetpw", data={"userName": "ghost", "email": "a@b.c"}
    )
    assert resp.json() == {"result": False}


# --- /setnewpw ---------------------------------------------------------------
def test_set_new_password_updates_credentials(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")
    app_client.post("/resetpw", data={"userName": "alice", "email": "a@b.c"})

    main = app_client.app_module
    with main.Session(main.engine) as session:
        code = main._get_user_by_name(session, "alice").activation_code

    resp = app_client.post(
        "/setnewpw",
        data={"userName": "alice", "newPassword": "newpw", "activationCode": code},
    )
    assert resp.json() == {"result": True}

    # Old password no longer works; new one does.
    bad = app_client.post("/login", data={"userName": "alice", "password": "pw"})
    assert bad.json()["result"] is False

    good = app_client.post(
        "/login", data={"userName": "alice", "password": "newpw"}
    )
    assert good.json() == {"result": True, "errorCode": 1}

    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert main._is_activated(user)


def test_set_new_password_wrong_activation_code_fails(app_client, helpers):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")
    app_client.post("/resetpw", data={"userName": "alice", "email": "a@b.c"})

    resp = app_client.post(
        "/setnewpw",
        data={
            "userName": "alice",
            "newPassword": "newpw",
            "activationCode": "000000",
        },
    )
    assert resp.json() == {"result": False}


def test_set_new_password_unknown_user_fails(app_client):
    resp = app_client.post(
        "/setnewpw",
        data={
            "userName": "ghost",
            "newPassword": "newpw",
            "activationCode": "abcdef",
        },
    )
    assert resp.json() == {"result": False}


# --- Full end-to-end flow ----------------------------------------------------
def test_full_user_lifecycle(app_client, helpers):
    """createUser -> activateUser -> login -> refreshLogin -> logout ->
    resetPassword -> setNewPassword -> login -> deleteUser."""
    main = app_client.app_module

    code = helpers.create_user(app_client, "alice", "pw", "a@b.c")
    assert helpers.activate_user(app_client, "alice", "pw", code=code).json() == {
        "result": True
    }

    assert app_client.post(
        "/login", data={"userName": "alice", "password": "pw"}
    ).json() == {"result": True, "errorCode": 1}

    assert app_client.post(
        "/refreshlogin", data={"userName": "alice", "password": "pw"}
    ).json() == {"result": True}

    assert app_client.post(
        "/logout", data={"userName": "alice", "password": "pw"}
    ).json() == {"result": True}

    assert app_client.post(
        "/resetpw", data={"userName": "alice", "email": "a@b.c"}
    ).json() == {"result": True}

    with main.Session(main.engine) as session:
        reset_code = main._get_user_by_name(session, "alice").activation_code

    assert app_client.post(
        "/setnewpw",
        data={
            "userName": "alice",
            "newPassword": "newpw",
            "activationCode": reset_code,
        },
    ).json() == {"result": True}

    assert app_client.post(
        "/login", data={"userName": "alice", "password": "newpw"}
    ).json() == {"result": True, "errorCode": 1}

    assert app_client.post(
        "/deleteuser", data={"userName": "alice", "password": "newpw"}
    ).json() == {"result": True}

    with main.Session(main.engine) as session:
        assert main._get_user_by_name(session, "alice") is None


# --- Activation email -------------------------------------------------------
def test_create_user_sends_activation_email(app_client, monkeypatch):
    main = app_client.app_module
    sent = []

    def fake_send(recipient, user_name, code):
        sent.append((recipient, user_name, code))
        return True

    monkeypatch.setattr(main, "_send_activation_email", fake_send)

    resp = app_client.post(
        "/createuser",
        data={"userName": "alice", "password": "pw", "email": " a@b.c "},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": True}

    assert len(sent) == 1
    recipient, user_name, code = sent[0]
    # Spaces in the email are stripped before delivery (matches old PHP).
    assert recipient == "a@b.c"
    assert user_name == "alice"
    with main.Session(main.engine) as session:
        user = main._get_user_by_name(session, "alice")
        assert user.activation_code == code


def test_create_user_succeeds_when_smtp_send_fails(app_client, monkeypatch):
    main = app_client.app_module
    monkeypatch.setattr(
        main, "_send_activation_email", lambda *_a, **_k: False
    )

    resp = app_client.post(
        "/createuser",
        data={"userName": "bob", "password": "pw", "email": "b@c.d"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": True}


def test_resetpw_sends_activation_email(app_client, helpers, monkeypatch):
    helpers.create_user(app_client, "alice", "pw", "a@b.c")
    helpers.activate_user(app_client, "alice", "pw")

    main = app_client.app_module
    sent = []
    monkeypatch.setattr(
        main,
        "_send_activation_email",
        lambda r, u, c: sent.append((r, u, c)) or True,
    )

    resp = app_client.post(
        "/resetpw", data={"userName": "alice", "email": "a@b.c"}
    )
    assert resp.json() == {"result": True}

    assert len(sent) == 1
    recipient, user_name, code = sent[0]
    assert recipient == "a@b.c"
    assert user_name == "alice"
    with main.Session(main.engine) as session:
        assert main._get_user_by_name(session, "alice").activation_code == code


def test_send_activation_email_skips_when_smtp_host_unset(app_client, monkeypatch):
    main = app_client.app_module
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert main._send_activation_email("a@b.c", "alice", "abcdef") is False
