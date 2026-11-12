"""End-to-end tests for the resource-management endpoints in main.py.

These cover NetworkService's getNetworkResourceList, getUserList,
getEmojiTypeByResourceId, getUserNamesForResourceAndEmojiType,
toggleReactionForResource, uploadResource, replaceResource, downloadResource,
incDownloadCounter, editResource, moveResource and deleteResource.
"""

from __future__ import annotations

import struct


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
        sim = session.get(main.Resource, sim_id)
        assert sim is not None
        assert sim.name == "alice-sim"
        assert sim.width == 32
        assert sim.height == 16
        assert sim.particles == 200
        assert sim.version == "2.0"
        assert sim.description == "desc"
        assert bytes(sim.content) == b"\x00\x01\x02payload"
        assert sim.size == len(b"\x00\x01\x02payload")
        assert sim.workspace == 0
        assert sim.type == 0
        assert bytes(sim.picture) == b""


def test_upload_simulation_stores_picture(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    picture = b"\xff\xd8\xff\xe0jpg-bytes\x00\xff\xd9"
    resp = helpers.upload_simulation(app_client, "alice", "pw", picture=picture)
    assert resp.json()["result"] is True

    sim_id = int(resp.json()["simId"])
    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Resource, sim_id)
        assert bytes(sim.picture) == picture


def test_upload_simulation_rejects_oversized_picture(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    main = app_client.app_module
    resp = helpers.upload_simulation(
        app_client, "alice", "pw", picture=b"x" * (main._MAX_PICTURE_SIZE + 1)
    )
    assert resp.status_code == 400


def _set_stored_size(app_client, sim_id: int, size: int):
    """Pretend the resource occupies ``size`` bytes without uploading them."""
    main = app_client.app_module
    with main.Session(main.engine) as session:
        with session.begin():
            session.get(main.Resource, sim_id).size = size


def _count_resources(app_client, user_name: str) -> int:
    main = app_client.app_module
    with main.Session(main.engine) as session:
        return len(
            session.execute(
                main.select(main.Resource.id)
                .join(main.User, main.User.id == main.Resource.user_id)
                .where(main.User.name == user_name)
            ).all()
        )


def test_upload_simulation_rejects_exceeded_storage_quota(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    main = app_client.app_module
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _set_stored_size(app_client, sim_id, main._MAX_TOTAL_SIZE_PER_USER)

    resp = helpers.upload_simulation(app_client, "alice", "pw", content=b"x")
    assert resp.json() == {"result": False}
    assert _count_resources(app_client, "alice") == 1


def test_upload_simulation_accepts_exactly_filled_storage_quota(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    main = app_client.app_module
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _set_stored_size(app_client, sim_id, main._MAX_TOTAL_SIZE_PER_USER - 3)

    resp = helpers.upload_simulation(app_client, "alice", "pw", content=b"abc")
    assert resp.json()["result"] is True


def test_upload_simulation_storage_quota_is_per_user(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    main = app_client.app_module
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _set_stored_size(app_client, sim_id, main._MAX_TOTAL_SIZE_PER_USER)

    resp = helpers.upload_simulation(app_client, "bob", "pw2", content=b"x")
    assert resp.json()["result"] is True


def _decode_pictures(content: bytes) -> dict[str, bytes]:
    """Decode the binary picture response, mirroring the C++ client."""
    pos = 0

    def read_block() -> bytes:
        nonlocal pos
        (size,) = struct.unpack_from("<I", content, pos)
        pos += 4
        block = content[pos : pos + size]
        pos += size
        return block

    (num_pictures,) = struct.unpack_from("<I", content, pos)
    pos += 4

    result = {}
    for _ in range(num_pictures):
        sim_id = read_block().decode("ascii")
        result[sim_id] = read_block()
    assert pos == len(content)
    return result


def test_get_simulation_pictures_returns_jpgs_for_requested_ids(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    picture = b"\xff\xd8\xff\xe0jpg-bytes\x00\xff\xd9"
    withPicture = helpers.upload_simulation(app_client, "alice", "pw", sim_name="a", picture=picture)
    withoutPicture = helpers.upload_simulation(app_client, "alice", "pw", sim_name="b")

    idWithPicture = withPicture.json()["simId"]
    idWithoutPicture = withoutPicture.json()["simId"]
    resp = app_client.post(
        "/getsimulationpictures", data={"simIds": idWithPicture + "," + idWithoutPicture}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    jpgById = _decode_pictures(resp.content)
    assert jpgById[idWithPicture] == picture
    assert jpgById[idWithoutPicture] == b""


def test_get_simulation_pictures_ignores_unknown_ids(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]

    resp = app_client.post("/getsimulationpictures", data={"simIds": sim_id + ",999999"})
    assert list(_decode_pictures(resp.content)) == [sim_id]


def test_get_simulation_pictures_rejects_too_many_ids(app_client, helpers):
    main = app_client.app_module
    simIds = ",".join(str(i) for i in range(main._MAX_PICTURES_PER_REQUEST + 1))
    resp = app_client.post("/getsimulationpictures", data={"simIds": simIds})
    assert resp.status_code == 400


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
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Resource, sim_id)
        assert bytes(sim.content) == b"NEW"
        assert sim.width == 100
        assert sim.height == 50
        assert sim.particles == 1234
        assert sim.version == "9.9"
        assert sim.size == 3


def test_replace_simulation_updates_picture(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw", picture=b"old-jpg").json()["simId"]
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
            "picture": ("picture.jpg", b"new-jpg", "image/jpeg"),
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        assert bytes(session.get(main.Resource, sim_id).picture) == b"new-jpg"


def test_replace_simulation_keeps_picture_when_none_is_sent(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw", picture=b"old-jpg").json()["simId"]
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
        },
    )
    assert resp.json() == {"result": True}

    main = app_client.app_module
    with main.Session(main.engine) as session:
        assert bytes(session.get(main.Resource, sim_id).picture) == b"old-jpg"


def test_replace_simulation_allows_owner_in_featured_workspace(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw").json()["simId"]
    )
    main = app_client.app_module
    with main.Session(main.engine) as session:
        with session.begin():
            session.get(main.Resource, sim_id).workspace = 1

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
        },
    )
    assert resp.json() == {"result": True}

    with main.Session(main.engine) as session:
        assert bytes(session.get(main.Resource, sim_id).content) == b"NEW"


def test_replace_simulation_ignores_size_of_replaced_resource(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    main = app_client.app_module
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _set_stored_size(app_client, sim_id, main._MAX_TOTAL_SIZE_PER_USER)

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
        },
    )
    assert resp.json() == {"result": True}

    with main.Session(main.engine) as session:
        assert session.get(main.Resource, sim_id).size == 3


def test_replace_simulation_rejects_exceeded_storage_quota(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    main = app_client.app_module
    filling_id = int(
        helpers.upload_simulation(app_client, "alice", "pw", sim_name="filling").json()["simId"]
    )
    sim_id = int(
        helpers.upload_simulation(app_client, "alice", "pw", sim_name="other").json()["simId"]
    )
    _set_stored_size(app_client, filling_id, main._MAX_TOTAL_SIZE_PER_USER)

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
        },
    )
    assert resp.json() == {"result": False}

    with main.Session(main.engine) as session:
        assert bytes(session.get(main.Resource, sim_id).content) != b"NEW"


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
        },
    )
    assert resp.json() == {"result": False}


# --- /downloadcontent --------------------------------------------------------
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

    # Each call to /downloadcontent with chunkIndex==0 increments num_downloads.
    main = app_client.app_module
    with main.Session(main.engine) as session:
        sim = session.get(main.Resource, sim_id)
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
            assert session.get(main.Resource, sim_id).num_downloads == expected


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
        sim = session.get(main.Resource, sim_id)
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
        sim = session.get(main.Resource, sim_id)
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
        assert session.get(main.Resource, sim_id).workspace == 2

    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "alice", "password": "pw",
            "simId": str(sim_id), "targetWorkspace": "0",
        },
    )
    assert resp.json() == {"result": True}
    with main.Session(main.engine) as session:
        assert session.get(main.Resource, sim_id).workspace == 0


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
        assert session.get(main.Resource, sim_id) is None
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


# --- /togglelikesimulationanonymously, /getlikedsimulationsanonymously ------
INSTALL_ID = "0123456789abcdef0123456789abcdef"
OTHER_INSTALL_ID = "fedcba9876543210fedcba9876543210"


def _toggle_anonymously(client, sim_id, like_type="1", install_id=INSTALL_ID):
    return client.post(
        "/togglelikesimulationanonymously",
        data={"installId": install_id, "simId": str(sim_id), "likeType": like_type},
    ).json()


def _liked_anonymously(client, install_id=INSTALL_ID):
    return client.post("/getlikedsimulationsanonymously", data={"installId": install_id}).json()


def test_toggle_like_anonymously_lifecycle(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])

    assert _toggle_anonymously(app_client, sim_id, "1") == {"result": True}
    assert _liked_anonymously(app_client) == [{"id": sim_id, "likeType": 1}]
    assert _liked_anonymously(app_client, OTHER_INSTALL_ID) == []

    assert _toggle_anonymously(app_client, sim_id, "2") == {"result": True}
    assert _liked_anonymously(app_client) == [{"id": sim_id, "likeType": 2}]

    assert _toggle_anonymously(app_client, sim_id, "2") == {"result": True}
    assert _liked_anonymously(app_client) == []


def test_toggle_like_anonymously_rejects_invalid_install_id(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])

    for install_id in ("short", "0123456789ABCDEF0123456789ABCDEF", "0123456789abcdef0123456789abcdeg", "0" * 65):
        assert _toggle_anonymously(app_client, sim_id, install_id=install_id) == {"result": False}
    assert _liked_anonymously(app_client, "short") == {"result": False}

    main = app_client.app_module
    from sqlalchemy import select
    with main.Session(main.engine) as session:
        assert session.execute(select(main.AnonymousLike)).first() is None


def test_toggle_like_anonymously_rejects_private_and_unknown_resources(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    private_id = int(helpers.upload_simulation(app_client, "alice", "pw", workspace=2).json()["simId"])

    assert _toggle_anonymously(app_client, private_id) == {"result": False}
    assert _toggle_anonymously(app_client, private_id + 1000) == {"result": False}
    assert _liked_anonymously(app_client) == []


def test_anonymous_likes_are_counted_with_account_likes(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "1"},
    )
    _toggle_anonymously(app_client, sim_id, "1")
    _toggle_anonymously(app_client, sim_id, "0", OTHER_INSTALL_ID)

    entry = app_client.post("/getversionedsimulationlist", data={}).json()[0]
    assert entry["likes"] == 3
    assert entry["likesByType"] == {"0": 1, "1": 2}

    by_name = {row["userName"]: row for row in app_client.post("/getuserlist", data={}).json()}
    assert by_name["alice"]["starsReceived"] == 3
    assert by_name["bob"]["starsGiven"] == 1

    names = app_client.post("/getuserlikes", data={"simId": str(sim_id), "likeType": "1"}).json()
    assert names == [{"userName": "bob"}]


def test_delete_simulation_removes_anonymous_likes(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _toggle_anonymously(app_client, sim_id)

    resp = app_client.post(
        "/deletesimulation",
        data={"userName": "alice", "password": "pw", "simId": str(sim_id)},
    )
    assert resp.json() == {"result": True}
    assert _liked_anonymously(app_client) == []


def test_delete_user_removes_anonymous_likes_on_own_resources(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _toggle_anonymously(app_client, sim_id)

    resp = app_client.post("/deleteuser", data={"userName": "alice", "password": "pw"})
    assert resp.json() == {"result": True}
    assert _liked_anonymously(app_client) == []


def test_toggle_like_anonymously_limits_toggles_per_caller(app_client, helpers, monkeypatch):
    monkeypatch.setattr(app_client.app_module, "ANONYMOUS_TOGGLES_PER_HOUR", 3)
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])

    results = [_toggle_anonymously(app_client, sim_id)["result"] for _ in range(4)]
    assert results == [True, True, True, False]


def test_toggle_like_rejects_invalid_like_types(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])

    for like_type in ("-1", "49"):
        assert _toggle_anonymously(app_client, sim_id, like_type) == {"result": False}
        assert app_client.post(
            "/togglelikesimulation",
            data={"userName": "alice", "password": "pw", "simId": str(sim_id), "likeType": like_type},
        ).json() == {"result": False}
    assert app_client.post("/getversionedsimulationlist", data={}).json()[0]["likesByType"] == {}


def test_toggle_like_anonymously_limits_new_likes_per_simulation(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    other_sim_id = int(helpers.upload_simulation(app_client, "alice", "pw", sim_name="other").json()["simId"])
    install_ids = [format(i, "032x") for i in range(1, 5)]

    results = [_toggle_anonymously(app_client, sim_id, install_id=install_id)["result"] for install_id in install_ids]
    assert results == [True, True, True, False]
    assert _toggle_anonymously(app_client, other_sim_id, install_id=install_ids[3]) == {"result": True}
    assert app_client.post("/getversionedsimulationlist", data={}).json()[0]["likes"] == 3


def test_anonymous_callers_are_grouped_by_ipv6_prefix(app_client):
    main = app_client.app_module
    assert main._caller_address("2001:db8:1:2::1", 48) == main._caller_address("2001:db8:1:ffff::2", 48)
    assert main._caller_address("2001:db8:1:2::1", 48) != main._caller_address("2001:db8:2:2::1", 48)
    assert main._caller_address("203.0.113.7", 48) == "203.0.113.7"


def test_account_toggle_takes_over_anonymous_like_of_same_install(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _toggle_anonymously(app_client, sim_id, "1")

    liked = app_client.post(
        "/getlikedsimulations", data={"userName": "bob", "password": "pw2", "installId": INSTALL_ID}
    ).json()
    assert liked == [{"id": sim_id, "likeType": 1}]

    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "2", "installId": INSTALL_ID},
    ).json() == {"result": True}
    entry = app_client.post("/getversionedsimulationlist", data={}).json()[0]
    assert entry["likes"] == 1
    assert entry["likesByType"] == {"2": 1}
    assert _liked_anonymously(app_client) == []

    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "2", "installId": INSTALL_ID},
    ).json() == {"result": True}
    assert app_client.post("/getversionedsimulationlist", data={}).json()[0]["likes"] == 0


def test_account_toggle_removes_anonymous_like_of_same_type(app_client, helpers):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    helpers.create_active_user(app_client, "bob", "pw2", "b@c.d")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _toggle_anonymously(app_client, sim_id, "1")

    assert app_client.post(
        "/togglelikesimulation",
        data={"userName": "bob", "password": "pw2", "simId": str(sim_id), "likeType": "1", "installId": INSTALL_ID},
    ).json() == {"result": True}
    assert app_client.post("/getversionedsimulationlist", data={}).json()[0]["likes"] == 0
    assert app_client.post("/getlikedsimulations", data={"userName": "bob", "password": "pw2"}).json() == []


def test_toggle_like_anonymously_limits_installs_per_caller(app_client, helpers, monkeypatch):
    monkeypatch.setattr(app_client.app_module, "ANONYMOUS_INSTALLS_PER_DAY", 2)
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    install_ids = [format(i, "032x") for i in range(1, 4)]

    results = [_toggle_anonymously(app_client, sim_id, install_id=install_id)["result"] for install_id in install_ids]
    assert results == [True, True, False]
    assert _toggle_anonymously(app_client, sim_id, install_id=install_ids[0]) == {"result": True}


def test_anonymous_callers_are_forgotten_a_day_after_their_last_request(app_client, helpers, monkeypatch):
    main = app_client.app_module
    now = [1_000_000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")
    sim_id = int(helpers.upload_simulation(app_client, "alice", "pw").json()["simId"])
    _toggle_anonymously(app_client, sim_id)
    assert len(main._anonymous_callers) == 1

    now[0] += main.LAST_DAY_SECONDS
    app_client.post("/getversionedsimulationlist", data={"countAsActive": "false"})
    assert len(main._anonymous_callers) == 1

    now[0] += 1
    app_client.post("/getversionedsimulationlist", data={"countAsActive": "false"})
    assert len(main._anonymous_callers) == 0


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


# --- Discord notifications ---------------------------------------------------
def _capture_discord_requests(main, monkeypatch):
    """Redirect the webhook call and return the list of captured requests."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook")

    class _Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    requests = []

    def _urlopen(request, timeout=None):
        requests.append(request)
        return _Response()

    monkeypatch.setattr(main.urllib.request, "urlopen", _urlopen)
    return requests


def test_discord_notification_attaches_picture(app_client, helpers, monkeypatch):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    main = app_client.app_module
    requests = _capture_discord_requests(main, monkeypatch)

    picture = b"\xff\xd8\xff\xe0jpg-bytes\x00\xff\xd9"
    assert helpers.upload_simulation(app_client, "alice", "pw", picture=picture).json()["result"] is True

    assert len(requests) == 1
    request = requests[0]
    assert request.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b'name="files[0]"; filename="preview.jpg"' in request.data
    assert picture in request.data
    assert b"attachment://preview.jpg" in request.data


def test_discord_notification_attaches_genome_picture(app_client, helpers, monkeypatch):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    main = app_client.app_module
    requests = _capture_discord_requests(main, monkeypatch)

    picture = b"\xff\xd8\xff\xe0jpg-bytes\x00\xff\xd9"
    assert helpers.upload_simulation(app_client, "alice", "pw", sim_type=1, picture=picture).json()["result"] is True

    assert len(requests) == 1
    request = requests[0]
    assert request.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b"New genome added to the database" in request.data
    assert picture in request.data
    assert b"attachment://preview.jpg" in request.data


def test_discord_notification_attaches_stored_picture_when_genome_becomes_public(app_client, helpers, monkeypatch):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    main = app_client.app_module
    requests = _capture_discord_requests(main, monkeypatch)

    picture = b"\xff\xd8\xff\xe0jpg-bytes\x00\xff\xd9"
    sim_id = helpers.upload_simulation(app_client, "alice", "pw", sim_type=1, workspace=2, picture=picture).json()["simId"]
    assert requests == []

    resp = app_client.post(
        "/movesimulation",
        data={
            "userName": "alice", "password": "pw",
            "simId": sim_id, "targetWorkspace": "0",
        },
    )
    assert resp.json() == {"result": True}

    assert len(requests) == 1
    request = requests[0]
    assert b"New genome added to the database" in request.data
    assert picture in request.data
    assert b"attachment://preview.jpg" in request.data


def test_discord_notification_without_picture_stays_json(app_client, helpers, monkeypatch):
    helpers.create_active_user(app_client, "alice", "pw", "a@b.c")

    main = app_client.app_module
    requests = _capture_discord_requests(main, monkeypatch)

    assert helpers.upload_simulation(app_client, "alice", "pw").json()["result"] is True

    assert len(requests) == 1
    request = requests[0]
    assert request.get_header("Content-type") == "application/json"
    assert b"attachment://" not in request.data
