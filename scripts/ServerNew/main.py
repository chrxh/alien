import hashlib
import hmac
import os
import secrets

from fastapi import FastAPI, Form
from sqlalchemy import (
    create_engine,
    String,
    Integer,
    DateTime,
    delete,
    func,
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

    # Cumulative number of refreshLogin calls (each call adds 1), matching the
    # old PHP server's `TIME_SPENT = COALESCE(TIME_SPENT + 1, 1)` behavior.
    time_spent: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # GPU model reported by the client on login (used by the old PHP server for
    # Discord notifications; kept for parity but not used for notifications).
    gpu: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="ck_users_name_nonempty"),
    )

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
    # Email is stripped of spaces (matches old PHP behavior).
    normalized = email.replace(" ", "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _new_salt() -> str:
    # 32 hex chars (16 random bytes), fits in VARCHAR(64).
    return secrets.token_hex(16)


def _new_activation_code() -> str:
    # 6 hex characters, matching the old PHP server's bin2hex(random(3)).
    return secrets.token_hex(3)


def _is_activated(user: "User") -> bool:
    # The old server stored an empty string to mark an activated user.
    # The new schema uses NULL. Treat both as "activated" for safety.
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
    # Match the old PHP regex: non-empty, no spaces.
    return bool(user_name) and " " not in user_name


def _get_user_by_name(session: Session, user_name: str) -> "User | None":
    return session.execute(select(User).where(User.name == user_name)).scalar_one_or_none()


# --- API endpoints (form-encoded, matching the C++ NetworkService client) ---
@app.post("/createuser")
def create_user(
    userName: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
):
    if not _is_valid_user_name(userName):
        return {"result": False}

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

    # NOTE: The old PHP server used PHPMailer here to email the activation
    # code to the user. Email delivery is intentionally not configured for the
    # new server; the activation code is generated and persisted only.
    # Discord notifications are intentionally not sent.
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
    # errorCode mirrors the old PHP server:
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

            # The old PHP server also deleted dependent rows in `userlike` and
            # `simulation`; those tables do not yet exist in the new schema.
            session.execute(delete(User).where(User.name == userName))
    return {"result": True}


@app.post("/resetpw")
def reset_password(
    userName: str = Form(...),
    email: str = Form(...),
):
    email_hash = _hash_email(email)
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

    # NOTE: The old PHP server emailed the new activation code to the user.
    # Email delivery is intentionally not configured for the new server.
    # Discord notifications are intentionally not sent.
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
