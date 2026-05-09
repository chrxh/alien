"""End-to-end tests for the resource-management endpoints in main.py.

These cover NetworkService's getNetworkResources, getUserList,
getEmojiTypeByResourceId, getUserNamesForResourceAndEmojiType,
toggleReactionForResource, uploadResource, replaceResource, downloadResource,
incDownloadCounter, editResource, moveResource and deleteResource.
"""

from __future__ import annotations


# --- /uploadsimulation -------------------------------------------------------
def test_upload_simulation_persists_row_and_returns_sim_id(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    resp = helpers.upload_simulation(
        app_client,
        "alice",
        "pw",
        sim_name="alice-sim",
        description="desc",
        width=32,
        height=16,
        particles=200,
        version="2.0",
        content=b"\x00\x01\x02payload",
        settings='{"k":1}',
        statistics="stats-data",
        sim_type=0,
        workspace=0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"] is True
    assert body["simId"].isdigit()

    sim_id = int(body["simId"])
    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Simulation, sim_id)
        assert sim is not None
        assert sim.name == "alice-sim"
        assert sim.width == 32
        assert sim.height == 16
        assert sim.particles == 200
        assert sim.version == "2.0"
        assert sim.description == "desc"
        assert bytes(sim.content) == b"\x00\x01\x02payload"
        assert sim.settings == '{"k":1}'
        assert sim.statistics == "stats-data"
        assert sim.size == len(b"\x00\x01\x02payload")
        assert sim.workspace == 0
        assert sim.type == 0


def test_upload_simulation_wrong_password_fails(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    resp = helpers.upload_simulation(app_client, "alice", "wrong")
    assert resp.json() == {"result": False}


def test_upload_simulation_rejects_alien_project_workspace_for_normal_user(
    app_client, helpers
):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    resp = helpers.upload_simulation(app_client, "alice", "pw", workspace=1)
    assert resp.json() == {"result": False}


def test_upload_simulation_accepts_alien_project_workspace_for_alien_project_user(
    app_client, helpers
):
    helpers.create_active_user(app_client, "alien-project", "pw", "a@b.c")
    resp = helpers.upload_simulation(
        app_client, "alien-project", "pw", workspace=1
    )
    assert resp.json()["result"] is True


# --- /replacesimulation ------------------------------------------------------
def test_replace_simulation_updates_owner_resource(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/replacesimulation",
        files={
            "userName": (None, "alice"),
            "password": (None, "pw"),
            "simId": (None, str(sim_id)),
            "width": (None, "100"),
            "height": (None, "50"),
            "particles": (None, "1234"),
            "version": (None, "9.9"),
            "content": ("content.bin", b"NEW", "application/octet-stream"),
            "settings": (None, "new-settings"),
            "statistics": (None, "new-stats"),
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Simulation, sim_id)
        assert bytes(sim.content) == b"NEW"
        assert sim.width == 100
        assert sim.height == 50
        assert sim.particles == 1234
        assert sim.version == "9.9"
        assert sim.settings == "new-settings"
        assert sim.statistics == "new-stats"
        assert sim.size == 3


def test_replace_simulation_rejects_non_owner(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/replacesimulation",
        files={
            "userName": (None, "bob"),
            "password": (None, "pw2"),
            "simId": (None, str(sim_id)),
            "width": (None, "1"),
            "height": (None, "1"),
            "particles": (None, "1"),
            "version": (None, "v"),
            "content": ("content.bin", b"X", "application/octet-stream"),
            "settings": (None, "s"),
            "statistics": (None, "x"),
        },
    )
    assert resp.json() == {"result": False}


# --- /downloadcontent, /downloadsettings, /downloadstatistics ----------------
def test_download_endpoints_return_content_and_increment_counter(
    app_client, helpers
):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(
            app_client,
            "alice",
            "pw",
            content=b"raw-bytes",
            settings="my-settings",
            statistics="my-stats",
        ).json()["simId"]
    )

    resp = app_client.get("/downloadcontent", params={"id": str(sim_id)})
    assert resp.status_code == 200
    assert resp.content == b"raw-bytes"

    # chunkIndex>=1 returns empty body for client compatibility.
    resp = app_client.get(
        "/downloadcontent", params={"id": str(sim_id), "chunkIndex": "1"}
    )
    assert resp.status_code == 200
    assert resp.content == b""

    assert app_client.get(
        "/downloadsettings", params={"id": str(sim_id)}
    ).text == "my-settings"
    assert app_client.get(
        "/downloadstatistics", params={"id": str(sim_id)}
    ).text == "my-stats"

    # Each call to /downloadcontent with chunkIndex==0 increments num_downloads.
    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Simulation, sim_id)
        assert sim.num_downloads == 1


# --- /incdownloadcount -------------------------------------------------------
def test_inc_download_count_increments(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    main = app_client.app_module

    for expected in (1, 2, 3):
        assert app_client.get(
            "/incdownloadcount", params={"id": str(sim_id)}
        ).json() == {"result": True}
        with main.Session(main.engine) as session:
            assert session.get(main.Simulation, sim_id).num_downloads == expected


# --- /editsimulation ---------------------------------------------------------
def test_edit_simulation_updates_name_and_description(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(
            app_client, "alice", "pw", sim_name="orig", description="orig-desc"
        ).json()["simId"]
    )
    resp = app_client.post(
        "/editsimulation",
        data={
            "userName": "alice",
            "password": "pw",
            "simId": str(sim_id),
            "newName": "renamed",
            "newDescription": "renamed-desc",
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Simulation, sim_id)
        assert sim.name == "renamed"
        assert sim.description == "renamed-desc"


def test_edit_simulation_accepts_empty_description(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(
            app_client, "alice", "pw", sim_name="orig", description="orig-desc"
        ).json()["simId"]
    )
    resp = app_client.post(
        "/editsimulation",
        data={
            "userName": "alice",
            "password": "pw",
            "simId": str(sim_id),
            "newName": "renamed",
            "newDescription": "",
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Simulation, sim_id)
        assert sim.name == "renamed"
        assert sim.description == ""


def test_edit_simulation_rejects_non_owner(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/editsimulation",
        data={
            "userName": "bob",
            "password": "pw2",
            "simId": str(sim_id),
            "newName": "x",
            "newDescription": "y",
        },
    )
    assert resp.json() == {"result": False}


# --- /movesimulation --------------------------------------------------------
def test_move_simulation_to_private_and_back(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw", workspace=0).json()["simId"]
    )

    main = app_client.app_module

    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "alice", "password": "pw",
            "simId": str(sim_id), "targetWorkspace": "2",
        },
    )
    assert resp.json() == {"result": True}
    with main.Session(main.engine) as session:
        assert session.get(main.Simulation, sim_id).workspace == 2

    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "alice", "password": "pw",
            "simId": str(sim_id), "targetWorkspace": "0",
        },
    )
    assert resp.json() == {"result": True}
    with main.Session(main.engine) as session:
        assert session.get(main.Simulation, sim_id).workspace == 0


def test_move_simulation_rejects_alien_project_workspace(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "alice", "password": "pw",
            "simId": str(sim_id), "targetWorkspace": "1",
        },
    )
    assert resp.json() == {"result": False}


def test_move_simulation_rejects_non_owner(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "bob", "password": "pw2",
            "simId": str(sim_id), "targetWorkspace": "2",
        },
    )
    assert resp.json() == {"result": False}


# --- /deletesimulation ------------------------------------------------------
def test_delete_simulation_removes_row_and_dependent_likes(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    # Bob likes Alice's sim.
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "1"},
    ).json() == {"result": True}

    resp = app_client.post(
        "/deletesimulation",
        data={"userName": "alice", "password": "pw", "simId": str(sim_id)},
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    from sqlalchemy import select
    with main.Session(main.engine) as session:
        assert session.get(main.Simulation, sim_id) is None
        assert session.execute(
            select(main.UserLike).where(main.UserLike.simulation_id == sim_id)
        ).first() is None


def test_delete_simulation_rejects_non_owner(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    resp = app_client.post(
        "/deletesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id)},
    )
    assert resp.json() == {"result": False}


# --- /togglelikesimulation, /getlikedsimulations, /getuserlikes -------------
def test_toggle_like_lifecycle(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )

    # Bob adds reaction type 1.
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "1"},
    ).json() == {"result": True}

    liked = app_client.post(
        "/getlikedsimulations",
        data={"userName": "bob", "password": "pw2"},
    ).json()
    assert liked == [{"id": sim_id, "likeType": 1}]

    users_react_1 = app_client.post(
        "/getuserlikes",
        data={"simId": str(sim_id), "likeType": "1"},
    ).json()
    assert users_react_1 == [{"userName": "bob"}]

    # Switch to type 2.
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "2"},
    ).json() == {"result": True}
    liked = app_client.post(
        "/getlikedsimulations",
        data={"userName": "bob", "password": "pw2"},
    ).json()
    assert liked == [{"id": sim_id, "likeType": 2}]

    # Toggle off (same type as current).
    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "2"},
    ).json() == {"result": True}
    liked = app_client.post(
        "/getlikedsimulations",
        data={"userName": "bob", "password": "pw2"},
    ).json()
    assert liked == []


def test_get_user_likes_no_reaction_filter_returns_all(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    helpers.create_active_user(app_client, "carol", "pw3", "c@d.e")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    for u, p, t in (("bob", "pw2", "1"), ("carol", "pw3", "2")):
        assert app_client.post(
            "/togglelikesimulation",
            data={"userName": u, "password": p, "simId": str(sim_id), "likeType": t},
        ).json() == {"result": True}

    rows = app_client.post(
        "/getuserlikes", data={"simId": str(sim_id)}
    ).json()
    user_names = {r["userName"] for r in rows}
    assert user_names == {"bob", "carol"}


# --- /getversionedsimulationlist --------------------------------------------
def test_get_versioned_simulation_list_returns_public_entries(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.upload_simulation(
        app_client, "alice", "pw", sim_name="public-sim", workspace=0
    )

    resp = app_client.post(
        "/getversionedsimulationlist", data={"version": "1.0"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    entry = body[0]
    assert entry["simulationName"] == "public-sim"
    assert entry["userName"] == "alice"
    assert entry["workspace"] == 0
    assert entry["likes"] == 0
    assert entry["likesByType"] == {}
    assert isinstance(entry["timestamp"], str)
    assert entry["contentSize"].isdigit()


def test_get_versioned_simulation_list_filters_private_for_other_users(
    app_client, helpers
):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")

    helpers.upload_simulation(
        app_client, "alice", "pw", sim_name="alice-private", workspace=2
    )
    helpers.upload_simulation(
        app_client, "bob", "pw2", sim_name="bob-public", workspace=0
    )

    # Anonymous: only the public entry.
    resp = app_client.post("/getversionedsimulationlist", data={})
    names = [e["simulationName"] for e in resp.json()]
    assert names == ["bob-public"]

    # Bob (other user): also only the public entry.
    resp = app_client.post(
        "/getversionedsimulationlist",
        data={"userName": "bob", "password": "pw2"},
    )
    names = [e["simulationName"] for e in resp.json()]
    assert names == ["bob-public"]

    # Alice: sees her own private entry too.
    resp = app_client.post(
        "/getversionedsimulationlist",
        data={"userName": "alice", "password": "pw"},
    )
    names = sorted(e["simulationName"] for e in resp.json())
    assert names == ["alice-private", "bob-public"]


def test_get_versioned_simulation_list_aggregates_likes(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    helpers.create_active_user(app_client, "carol", "pw3", "c@d.e")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    for u, p, t in (("bob", "pw2", "1"), ("carol", "pw3", "1")):
        app_client.post(
            "/togglelikesimulation",
            data={"userName": u, "password": p, "simId": str(sim_id), "likeType": t},
        )

    body = app_client.post("/getversionedsimulationlist", data={}).json()
    assert len(body) == 1
    entry = body[0]
    assert entry["likes"] == 2
    assert entry["likesByType"] == {"1": 2}


# --- /getuserlist -----------------------------------------------------------
def test_get_user_list_excludes_pending_users_and_reports_stars(app_client, helpers):
    # Pending (un-activated) user must NOT appear.
    helpers.create_user(app_client, "pending", "pw", "p@p.p")

    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "1"},
    )

    body = app_client.post("/getuserlist", data={}).json()
    by_name = {row["userName"]: row for row in body}
    assert set(by_name) == {"alice", "bob"}
    assert by_name["alice"]["starsReceived"] == 1
    assert by_name["alice"]["starsGiven"] == 0
    assert by_name["bob"]["starsReceived"] == 0
    assert by_name["bob"]["starsGiven"] == 1
