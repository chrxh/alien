import os
from fastapi import FastAPI
from sqlalchemy import (
    create_engine,
    BigInteger,
    String,
    LargeBinary,
    Integer,
    DateTime,
    func,
    CheckConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# --- DB connection ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alien:CHANGE_ME@127.0.0.1:5432/alien_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- ORM base + model ---
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    name: Mapped[str] = mapped_column(String(128), nullable=False)

    pw_hash: Mapped[bytes] = mapped_column(LargeBinary(256), nullable=False)
    email_hash: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False, unique=True)
    salt: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)

    activation_code: Mapped[str | None] = mapped_column(String(128), nullable=True)

    flags: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    timestamp: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

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
