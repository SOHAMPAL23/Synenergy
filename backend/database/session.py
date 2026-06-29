"""
EnerVision AI - Database Engine & Session
Async SQLAlchemy setup for NeonDB (PostgreSQL).
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import settings


# ─── Base ORM class ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── Engine ───────────────────────────────────────────────────────────────────

engine_kwargs = {
    "echo": settings.DB_ECHO,
}

if settings.DATABASE_URL.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    engine_kwargs["poolclass"] = StaticPool
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    engine_kwargs["pool_pre_ping"] = True
    
    # Enable SSL for remote databases (exclude local hosts and the internal docker db host)
    db_host = settings.DATABASE_URL.split("@")[-1].split("/")[0].split(":")[0]
    if db_host not in ("localhost", "127.0.0.1", "db"):
        engine_kwargs["connect_args"] = {"ssl": "require"}

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ─── Dependency ───────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── Startup / Teardown helpers ───────────────────────────────────────────────

async def create_all_tables() -> None:
    """Create all tables defined via ORM models (used in dev/test) and seed default admin user."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed system admin user if not exists
    from backend.repositories import UserRepository
    from backend.models.orm import User
    from backend.core.security import hash_password

    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)
        existing_admin = await repo.get_by_email(settings.FIRST_SUPERUSER_EMAIL)
        if not existing_admin:
            admin_user = User(
                email=settings.FIRST_SUPERUSER_EMAIL,
                hashed_password=hash_password(settings.FIRST_SUPERUSER_PASSWORD),
                full_name="System Administrator",
                role="admin",
                is_active=True,
                is_verified=True,
            )
            await repo.create(admin_user)
            await db.commit()


async def drop_all_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
