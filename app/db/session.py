from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.settings import settings
from app.db import base

try:  # pragma: no cover - optional dependency
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None
engine = create_engine(str(settings.DATABASE_URL), future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_connection():
    host = base.HOST or "localhost"
    port = base.PORT or "5432"
    name = base.NAME
    user = base.USER
    password = base.PASSWORD

    missing = []
    if not name:
        missing.append("POSTGRES_DB")
    if not user:
        missing.append("POSTGRES_USER")
    if not password:
        missing.append("POSTGRES_PASSWORD")

    if missing:
        raise ValueError(
            "Missing database configuration: " + ", ".join(missing)
        )
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for direct DB connections")
    return psycopg2.connect(
        # host=base.server,
        # dbname=base.name,
        # user=base.user,
        # password=base.pw,
        # port=base.port,
        host=host,
        dbname=name,
        user=user,
        password=password,
        port=port,
    )

