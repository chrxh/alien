import hashlib
import hmac
import logging
import os
import secrets
import smtplib
from email.message import EmailMessage

from fastapi import FastAPI, Form
from sqlalchemy import (
    create_engine,
    String,
    Integer,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    LargeBinary,
    Text,
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

    from_release: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0"
    )
    size: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default="0"
    )

    type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    statistics: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    _send_activation_email(normalized_email, userName, activation_code)
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

            session.execute(delete(User).where(User.name == userName))
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
