import hashlib
import hmac
import logging
import os
import secrets
import smtplib
from email.message import EmailMessage

from fastapi import FastAPI, Form, Response
from sqlalchemy import (
    create_engine,
    String,
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    LargeBinary,
    Text,
    and_,
    delete,
    func,
    or_,
    select,
    update,
    CheckConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# --- DB connection ---
# DATABASE_URL is required in production. We deliberately do not ship a default
# password here so that mis-configured deployments fail fast with a clear error
# instead of silently trying a placeholder credential. The test suite injects
# its own DATABASE_URL via the ``app_client`` fixture in tests/conftest.py.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Configure it before starting the server, e.g.\n"
        "  export DATABASE_URL='postgresql+psycopg://alien:<password>@127.0.0.1:5432/alien_db'\n"
        "or set EnvironmentFile=/etc/alien-server.env in the systemd unit."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- ORM base + model ---
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    pw_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="SHA-256 hash of password with salt"
    )
    email_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="SHA-256 hash of email with salt"
    )
    salt: Mapped[str] = mapped_column(String(64), nullable=False, comment="Salt")

    activation_code: Mapped[str | None] = mapped_column(String(6), nullable=True)

    flags: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    timestamp: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last activity",
    )

    # Cumulative number of refreshLogin calls (each call adds 1)
    time_spent: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # GPU model reported by the client on login
    gpu: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="ck_users_name_nonempty"),
    )


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    particles: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    settings: Mapped[str] = mapped_column(Text, nullable=False)
    picture: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    num_downloads: Mapped[int] = mapped_column(Integer, nullable=False)

    timestamp: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Workspace type for the simulation:
    #   0 -> Public, 1 -> AlienProject, 2 -> Private (see source/Network/Definitions.h).
    workspace: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    size: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default="0"
    )

    type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    statistics: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserLike(Base):
    __tablename__ = "userlikes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id"), nullable=False
    )

    type: Mapped[int | None] = mapped_column(Integer, nullable=True)


# --- FastAPI app ---
app = FastAPI()

@app.on_event("startup")
def startup():
    # creates tables only if they don't exist
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}


# --- Helpers ---
def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(password.encode("utf-8") + salt.encode("utf-8")).hexdigest()


def _hash_email(email: str) -> str:
    # Email is stripped of spaces
    normalized = email.replace(" ", "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _new_salt() -> str:
    # 32 hex chars (16 random bytes), fits in VARCHAR(64).
    return secrets.token_hex(16)


def _new_activation_code() -> str:
    # 6 hex characters
    return secrets.token_hex(3)


# --- Email delivery ----------------------------------------------------------
# SMTP configuration, sourced from environment variables. Email delivery is
# optional: if SMTP_HOST is not set, the server logs a warning and skips
# sending. This keeps the endpoint usable in dev/CI without an SMTP server,
# while still sending real activation emails in production.
#
# In production set (e.g. via /etc/alien-server.env):
#   SMTP_HOST=alfa3211.alfahosting-server.de
#   SMTP_PORT=465
#   SMTP_USE_SSL=true
#   SMTP_USERNAME=<smtp user>
#   SMTP_PASSWORD=<smtp pass>
#   SMTP_FROM="User registration <info@alien-project.org>"
_SMTP_FROM_DEFAULT = "User registration <info@alien-project.org>"

_logger = logging.getLogger("alien-server.email")


def _send_activation_email(recipient: str, user_name: str, activation_code: str) -> bool:
    """Send the activation code to ``recipient``.

    Returns True if the message was handed off to the SMTP server, False if
    SMTP is not configured or delivery failed. Failures are logged but never
    raised: the caller should still succeed even when email delivery is
    unavailable.
    """
    host = os.getenv("SMTP_HOST")
    if not host:
        _logger.warning(
            "SMTP_HOST not set; skipping activation email for user %r", user_name
        )
        return False

    port = int(os.getenv("SMTP_PORT", "465"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_ssl = os.getenv("SMTP_USE_SSL", "true").lower() in ("1", "true", "yes")
    sender = os.getenv("SMTP_FROM", _SMTP_FROM_DEFAULT)

    msg = EmailMessage()
    msg["Subject"] = (
        f"Artificial Life Environment: confirmation code for user '{user_name}'"
    )
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        f"Your confirmation code for user '{user_name}' is:\n\n{activation_code}"
    )

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.starttls()
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        return True
    except (OSError, smtplib.SMTPException) as exc:
        _logger.error("Failed to send activation email to %r: %s", recipient, exc)
        return False


def _is_activated(user: "User") -> bool:
    # NULL to mark an activated user.
    return not user.activation_code


def _check_password(user: "User", password: str) -> bool:
    expected = _hash_password(password, user.salt)
    return hmac.compare_digest(expected, user.pw_hash) and _is_activated(user)


def _check_password_and_activation_code(user: "User", password: str, activation_code: str) -> bool:
    expected = _hash_password(password, user.salt)
    return (
        hmac.compare_digest(expected, user.pw_hash)
        and (user.activation_code or "") == activation_code
    )


def _is_valid_user_name(user_name: str) -> bool:
    return bool(user_name) and " " not in user_name


def _get_user_by_name(session: Session, user_name: str) -> "User | None":
    return session.execute(select(User).where(User.name == user_name)).scalar_one_or_none()


# --- Workspace types (matching source/Network/Definitions.h) ---
_WORKSPACE_PUBLIC = 0
_WORKSPACE_ALIEN_PROJECT = 1
_WORKSPACE_PRIVATE = 2

# Special user name allowed to publish into the curated alien-project workspace.
_ALIEN_PROJECT_USER_NAME = "alien-project"


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _checked_user(session: Session, user_name: str, password: str) -> "User | None":
    """Return the user iff name+password are valid (and account is activated)."""
    user = _get_user_by_name(session, user_name)
    if user is None or not _check_password(user, password):
        return None
    return user


# --- API endpoints (form-encoded, matching the C++ NetworkService client) ---
@app.post("/createuser")
def create_user(
    userName: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
):
    if not _is_valid_user_name(userName):
        return {"result": False}

    # Stripped spaces from the email before hashing AND before
    # using it as the SMTP recipient.
    normalized_email = email.replace(" ", "")

    salt = _new_salt()
    pw_hash = _hash_password(password, salt)
    email_hash = _hash_email(email)
    activation_code = _new_activation_code()

    with Session(engine) as session:
        with session.begin():
            existing = _get_user_by_name(session, userName)
            if existing is not None:
                # If the previous record is still pending activation, replace it.
                # Otherwise the user name is already taken.
                if _is_activated(existing):
                    return {"result": False}
                session.execute(delete(User).where(User.name == userName))

            session.add(
                User(
                    name=userName,
                    pw_hash=pw_hash,
                    email_hash=email_hash,
                    salt=salt,
                    activation_code=activation_code,
                    flags=0,
                )
            )

    # Send the activation code to the user's email address.
    if not _send_activation_email(normalized_email, userName, activation_code):
        return {"result": False}
    return {"result": True}


@app.post("/activateuser")
def activate_user(
    userName: str = Form(...),
    password: str = Form(...),
    activationCode: str = Form(...),
    gpu: str | None = Form(None),
):
    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or not _check_password_and_activation_code(user, password, activationCode):
                return {"result": False}

            session.execute(
                update(User).where(User.name == userName).values(activation_code=None)
            )

    # Discord notifications are intentionally not sent (the `gpu` parameter is
    # accepted for client compatibility but ignored).
    _ = gpu
    return {"result": True}


@app.post("/login")
def login(
    userName: str = Form(...),
    password: str = Form(...),
    gpu: str | None = Form(None),
):
    # errorCode:
    #   0 -> user exists but is not activated yet
    #   1 -> default (e.g. unknown user / wrong password)
    error_code = 1
    success = False

    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is not None and not _is_activated(user):
                error_code = 0

            if user is not None and _check_password(user, password):
                session.execute(
                    update(User)
                    .where(User.name == userName)
                    .values(flags=1, timestamp=func.now(), gpu=(gpu or ""))
                )
                success = True

    # Discord notifications are intentionally not sent.
    return {"result": success, "errorCode": error_code}


@app.post("/logout")
def logout(
    userName: str = Form(...),
    password: str = Form(...),
):
    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or not _check_password(user, password):
                return {"result": False}

            session.execute(
                update(User)
                .where(User.name == userName)
                .values(flags=0, timestamp=func.now())
            )
    return {"result": True}


@app.post("/refreshlogin")
def refresh_login(
    userName: str = Form(...),
    password: str = Form(...),
):
    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or not _check_password(user, password):
                return {"result": False}

            session.execute(
                update(User)
                .where(User.name == userName)
                .values(
                    flags=1,
                    timestamp=func.now(),
                    time_spent=func.coalesce(User.time_spent + 1, 1),
                )
            )
    return {"result": True}


@app.post("/deleteuser")
def delete_user(
    userName: str = Form(...),
    password: str = Form(...),
):
    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or not _check_password(user, password):
                return {"result": False}

            # Delete dependent rows first (foreign keys to "users"):
            #   - userlikes:    references the user directly via user_id
            #   - userlikes:    references the user's simulations via simulation_id
            #   - simulations:  references the user via user_id
            user_id = user.id
            user_sim_ids = list(
                session.execute(
                    select(Simulation.id).where(Simulation.user_id == user_id)
                ).scalars()
            )
            session.execute(delete(UserLike).where(UserLike.user_id == user_id))
            if user_sim_ids:
                session.execute(
                    delete(UserLike).where(UserLike.simulation_id.in_(user_sim_ids))
                )
            session.execute(delete(Simulation).where(Simulation.user_id == user_id))
            session.execute(delete(User).where(User.id == user_id))
    return {"result": True}


@app.post("/resetpw")
def reset_password(
    userName: str = Form(...),
    email: str = Form(...),
):
    email_hash = _hash_email(email)
    normalized_email = email.replace(" ", "")
    activation_code = _new_activation_code()

    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or not hmac.compare_digest(user.email_hash, email_hash):
                return {"result": False}

            session.execute(
                update(User)
                .where(User.name == userName)
                .values(activation_code=activation_code)
            )

    # Email the new activation code. Best-effort:
    # we still report success on SMTP failure so the caller's flow isn't broken
    # by transient mail-server issues.
    _send_activation_email(normalized_email, userName, activation_code)
    return {"result": True}


@app.post("/setnewpw")
def set_new_password(
    userName: str = Form(...),
    newPassword: str = Form(...),
    activationCode: str = Form(...),
):
    with Session(engine) as session:
        with session.begin():
            user = _get_user_by_name(session, userName)
            if user is None or (user.activation_code or "") != activationCode:
                return {"result": False}

            salt = _new_salt()
            pw_hash = _hash_password(newPassword, salt)
            session.execute(
                update(User)
                .where(User.name == userName)
                .values(pw_hash=pw_hash, salt=salt, activation_code=None)
            )
    return {"result": True}


# --- Resource endpoints ------------------------------------------------------
# All public resource queries are anonymous; private resources are filtered to
# the calling user via userName/password (when supplied).


@app.post("/getversionedsimulationlist")
def get_versioned_simulation_list(
    userName: str | None = Form(None),
    password: str | None = Form(None),
    version: str | None = Form(None),
):
    """Return all visible simulations (the C++ getNetworkResources endpoint).

    Private (workspace=2) entries are only returned to their owner. The
    ``version`` parameter is accepted for client compatibility and currently
    not used to filter results.
    """
    _ = version

    with Session(engine) as session:
        # Aggregate likes by (simulation_id, type).
        likes_rows = session.execute(
            select(UserLike.simulation_id, UserLike.type, func.count())
            .group_by(UserLike.simulation_id, UserLike.type)
        ).all()
        likes_by_sim: dict[int, dict[int, int]] = {}
        for sim_id, like_type, num in likes_rows:
            normalized_type = 0 if like_type is None else int(like_type)
            bucket = likes_by_sim.setdefault(sim_id, {})
            bucket[normalized_type] = bucket.get(normalized_type, 0) + int(num)

        # Determine whether the caller is authenticated (for private filter).
        caller_name: str | None = None
        if userName and password:
            caller = _get_user_by_name(session, userName)
            if caller is not None and _check_password(caller, password):
                caller_name = userName

        sims = session.execute(
            select(
                Simulation.id,
                Simulation.name,
                Simulation.description,
                User.name,
                Simulation.width,
                Simulation.height,
                Simulation.particles,
                Simulation.version,
                Simulation.timestamp,
                Simulation.num_downloads,
                Simulation.size,
                Simulation.workspace,
                Simulation.type,
            ).join(User, User.id == Simulation.user_id, isouter=True)
        ).all()

        result = []
        for (
            sim_id,
            sim_name,
            description,
            owner_name,
            width,
            height,
            particles,
            sim_version,
            timestamp,
            num_downloads,
            size,
            workspace,
            sim_type,
        ) in sims:
            if int(workspace) == _WORKSPACE_PRIVATE and (
                caller_name is None or owner_name != caller_name
            ):
                continue

            likes_by_type = likes_by_sim.get(sim_id, {})
            total_likes = sum(likes_by_type.values())
            result.append(
                {
                    "id": int(sim_id),
                    "simulationName": sim_name or "",
                    "userName": owner_name or "",
                    "description": description or "",
                    "width": int(width),
                    "height": int(height),
                    "particles": int(particles),
                    "version": sim_version or "",
                    # The C++ client parses ``timestamp`` and ``contentSize``
                    # as strings (see NetworkResourceParserService.cpp).
                    "timestamp": "" if timestamp is None else str(timestamp),
                    "contentSize": str(int(size)),
                    "likes": int(total_likes),
                    # boost::property_tree expects a JSON object here; emit
                    # string-keyed entries so the parser maps them by like type.
                    "likesByType": {str(k): v for k, v in likes_by_type.items()},
                    "numDownloads": int(num_downloads),
                    "workspace": int(workspace),
                    "type": 0 if sim_type is None else int(sim_type),
                }
            )
        return result


@app.post("/getuserlist")
def get_user_list():
    with Session(engine) as session:
        # Stars received: total userlikes on simulations owned by each user.
        stars_received_rows = session.execute(
            select(User.id, func.count())
            .select_from(User)
            .join(Simulation, Simulation.user_id == User.id)
            .join(UserLike, UserLike.simulation_id == Simulation.id)
            .group_by(User.id)
        ).all()
        stars_received_by_user = {uid: int(n) for uid, n in stars_received_rows}

        # Stars given: userlikes authored by each user.
        likes_given_rows = session.execute(
            select(UserLike.user_id, func.count()).group_by(UserLike.user_id)
        ).all()
        likes_given_by_user = {uid: int(n) for uid, n in likes_given_rows}

        users = session.execute(
            select(User.id, User.name, User.timestamp, User.time_spent, User.gpu, User.flags)
            .where(User.activation_code.is_(None))
        ).all()

        # "Online" = active in the last 60 minutes AND flags=1.
        # "Last day online" = active in the last 24h.
        # Time arithmetic is done in Python so it works identically on
        # Postgres and SQLite, regardless of dialect-specific INTERVAL syntax.
        from datetime import datetime, timedelta, timezone

        # Compute "now" from the DB when possible so the cut-off uses the same
        # clock as the stored timestamps; fall back to wall-clock time.
        try:
            db_now = session.execute(select(func.now())).scalar_one()
        except Exception:
            db_now = datetime.now(timezone.utc)
        if db_now is not None and db_now.tzinfo is None:
            db_now = db_now.replace(tzinfo=timezone.utc)

        result = []
        for uid, name, ts, time_spent, gpu, flags in users:
            # Make the timestamp timezone-aware for comparison; SQLite returns
            # naive datetimes even for ``DateTime(timezone=True)`` columns.
            if ts is not None and ts.tzinfo is None:
                ts_aware = ts.replace(tzinfo=timezone.utc)
            else:
                ts_aware = ts

            online = bool(
                flags == 1
                and ts_aware is not None
                and (db_now - ts_aware) <= timedelta(minutes=60)
            )
            last_day_online = bool(
                ts_aware is not None and (db_now - ts_aware) <= timedelta(hours=24)
            )
            result.append(
                {
                    "userName": name or "",
                    "starsReceived": int(stars_received_by_user.get(uid, 0)),
                    "starsGiven": int(likes_given_by_user.get(uid, 0)),
                    "timestamp": "" if ts is None else str(ts),
                    "online": online,
                    "lastDayOnline": last_day_online,
                    "timeSpent": int(time_spent) if time_spent is not None else 0,
                    "gpu": gpu or "",
                }
            )
        return result


@app.post("/getlikedsimulations")
def get_liked_simulations(
    userName: str = Form(...),
    password: str = Form(...),
):
    with Session(engine) as session:
        user = _checked_user(session, userName, password)
        if user is None:
            return {"result": False}
        rows = session.execute(
            select(UserLike.simulation_id, UserLike.type).where(UserLike.user_id == user.id)
        ).all()
        return [
            {"id": int(sim_id), "likeType": 0 if t is None else int(t)}
            for sim_id, t in rows
        ]


@app.post("/getuserlikes")
def get_user_likes(
    simId: str = Form(...),
    likeType: str | None = Form(None),
):
    sim_id = _parse_int(simId)
    with Session(engine) as session:
        stmt = (
            select(User.name)
            .select_from(UserLike)
            .join(User, User.id == UserLike.user_id)
            .where(UserLike.simulation_id == sim_id)
        )
        if likeType is not None:
            lt = _parse_int(likeType)
            if lt == 0:
                # Match both explicit 0 and historical NULL entries.
                stmt = stmt.where(or_(UserLike.type == 0, UserLike.type.is_(None)))
            else:
                stmt = stmt.where(UserLike.type == lt)
        rows = session.execute(stmt).all()
        return [{"userName": name or ""} for (name,) in rows]


@app.post("/togglelikesimulation")
def toggle_like_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simId: str = Form(...),
    likeType: str | None = Form(None),
):
    sim_id = _parse_int(simId)
    new_like_type = _parse_int(likeType) if likeType is not None else 0

    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}

            existing = session.execute(
                select(UserLike).where(
                    and_(
                        UserLike.user_id == user.id,
                        UserLike.simulation_id == sim_id,
                    )
                )
            ).scalar_one_or_none()

            only_remove = False
            if existing is not None:
                orig_type = 0 if existing.type is None else int(existing.type)
                if orig_type == new_like_type:
                    only_remove = True
                session.execute(delete(UserLike).where(UserLike.id == existing.id))

            if not only_remove:
                session.add(
                    UserLike(
                        user_id=user.id,
                        simulation_id=sim_id,
                        type=new_like_type,
                    )
                )
        return {"result": True}


def _is_owner_of_simulation(session: Session, sim_id: int, user_name: str) -> bool:
    row = session.execute(
        select(User.name)
        .select_from(Simulation)
        .join(User, User.id == Simulation.user_id)
        .where(Simulation.id == sim_id)
    ).first()
    return row is not None and row[0] == user_name


@app.post("/uploadsimulation")
def upload_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simName: str = Form(...),
    simDesc: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    particles: int = Form(...),
    version: str = Form(...),
    content: bytes = Form(...),
    settings: str = Form(...),
    type: int = Form(0),
    workspace: int = Form(0),
    statistics: str = Form(""),
):
    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}

            # Only the dedicated ``alien-project`` user may publish into the
            # curated AlienProject workspace.
            if (
                workspace == _WORKSPACE_ALIEN_PROJECT
                and userName != _ALIEN_PROJECT_USER_NAME
            ):
                return {"result": False}

            sim = Simulation(
                user_id=user.id,
                name=simName,
                width=int(width),
                height=int(height),
                particles=int(particles),
                version=version,
                description=simDesc,
                content=content,
                settings=settings,
                picture=b"",
                num_downloads=0,
                workspace=int(workspace),
                size=len(content),
                type=int(type),
                statistics=statistics,
            )
            session.add(sim)
            session.flush()
            sim_id = sim.id

    return {"result": True, "simId": str(sim_id)}


@app.post("/replacesimulation")
def replace_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simId: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    particles: int = Form(...),
    version: str = Form(...),
    content: bytes = Form(...),
    settings: str = Form(...),
    statistics: str = Form(""),
):
    sim_id = _parse_int(simId)
    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}

            sim = session.get(Simulation, sim_id)
            if sim is None or sim.user_id != user.id:
                return {"result": False}

            # An entry in the curated workspace can only be replaced by the
            # dedicated alien-project user.
            if (
                sim.workspace == _WORKSPACE_ALIEN_PROJECT
                and userName != _ALIEN_PROJECT_USER_NAME
            ):
                return {"result": False}

            sim.particles = int(particles)
            sim.version = version
            sim.content = content
            sim.width = int(width)
            sim.height = int(height)
            sim.settings = settings
            sim.size = len(content)
            sim.statistics = statistics
    return {"result": True}


@app.get("/downloadcontent")
def download_content(id: str, chunkIndex: int = 0):
    """Return the simulation's CONTENT.

    Content is stored in a single column (no chunking). The ``chunkIndex``
    query parameter is accepted for client compatibility:
      * ``chunkIndex == 0`` (or absent): return the full content.
      * ``chunkIndex >= 1``: return an empty body.
    """
    sim_id = _parse_int(id)
    if chunkIndex >= 1:
        return Response(content=b"", media_type="application/octet-stream")

    with Session(engine) as session:
        with session.begin():
            sim = session.get(Simulation, sim_id)
            if sim is None:
                return Response(content=b"", media_type="application/octet-stream")
            # Increment the download counter without touching the timestamp.
            session.execute(
                update(Simulation)
                .where(Simulation.id == sim_id)
                .values(num_downloads=Simulation.num_downloads + 1, timestamp=Simulation.timestamp)
            )
            content = sim.content or b""
    return Response(content=bytes(content), media_type="application/octet-stream")


@app.get("/downloadsettings")
def download_settings(id: str):
    sim_id = _parse_int(id)
    with Session(engine) as session:
        sim = session.get(Simulation, sim_id)
        return Response(content=(sim.settings if sim is not None else "") or "", media_type="text/plain")


@app.get("/downloadstatistics")
def download_statistics(id: str):
    sim_id = _parse_int(id)
    with Session(engine) as session:
        sim = session.get(Simulation, sim_id)
        body = (sim.statistics if sim is not None else "") or ""
    return Response(content=body, media_type="text/plain")


@app.get("/incdownloadcount")
def inc_download_count(id: str):
    sim_id = _parse_int(id)
    with Session(engine) as session:
        with session.begin():
            session.execute(
                update(Simulation)
                .where(Simulation.id == sim_id)
                .values(num_downloads=Simulation.num_downloads + 1, timestamp=Simulation.timestamp)
            )
    return {"result": True}


@app.post("/editsimulation")
def edit_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simId: str = Form(...),
    newName: str = Form(...),
    newDescription: str = Form(...),
):
    sim_id = _parse_int(simId)
    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}
            if not _is_owner_of_simulation(session, sim_id, userName):
                return {"result": False}
            session.execute(
                update(Simulation)
                .where(Simulation.id == sim_id)
                .values(name=newName, description=newDescription, timestamp=Simulation.timestamp)
            )
    return {"result": True}


@app.post("/movesimulation")
def move_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simId: str = Form(...),
    targetWorkspace: str = Form(...),
):
    sim_id = _parse_int(simId)
    target = _parse_int(targetWorkspace)
    # A user can move only between Public (0) and Private (2); the curated
    # AlienProject (1) workspace is off-limits for normal moves.
    if target not in (_WORKSPACE_PUBLIC, _WORKSPACE_PRIVATE):
        return {"result": False}
    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}
            if not _is_owner_of_simulation(session, sim_id, userName):
                return {"result": False}
            session.execute(
                update(Simulation)
                .where(Simulation.id == sim_id)
                .values(workspace=target, timestamp=Simulation.timestamp)
            )
    return {"result": True}


@app.post("/deletesimulation")
def delete_simulation(
    userName: str = Form(...),
    password: str = Form(...),
    simId: str = Form(...),
):
    sim_id = _parse_int(simId)
    with Session(engine) as session:
        with session.begin():
            user = _checked_user(session, userName, password)
            if user is None:
                return {"result": False}
            # Only the owner may delete; an unknown ``simId`` is treated as
            # success (idempotent delete).
            row = session.execute(
                select(User.name)
                .select_from(Simulation)
                .join(User, User.id == Simulation.user_id)
                .where(Simulation.id == sim_id)
            ).first()
            if row is not None and row[0] != userName:
                return {"result": False}

            session.execute(delete(UserLike).where(UserLike.simulation_id == sim_id))
            session.execute(delete(Simulation).where(Simulation.id == sim_id))
    return {"result": True}
