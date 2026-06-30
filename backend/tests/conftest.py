# Force testing configuration before loading app or database modules
from backend.core.config import settings
settings.ENVIRONMENT = "testing"
settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"

import asyncio
import uuid
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import Base, get_db, engine as test_engine, AsyncSessionLocal as TestSessionLocal
from backend.main import app
from backend.core.security import create_access_token, hash_password


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator:
    """Provide a shared event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_test_tables():
    """Create all tables in the test database once per session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a fresh DB session per test, rolled back after each test."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTPX test client with the test DB injected."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Helper fixtures ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def viewer_token(db_session: AsyncSession) -> str:
    """Register a viewer user and return a valid JWT access token."""
    from backend.models.orm import User

    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email=f"viewer_{user_id.hex[:8]}@test.com",
        hashed_password=hash_password("TestPass123"),
        full_name="Test Viewer",
        role="viewer",
    )
    db_session.add(user)
    await db_session.flush()

    return create_access_token(str(user_id), {"role": "viewer"})


@pytest_asyncio.fixture
async def analyst_token(db_session: AsyncSession) -> str:
    """Register an analyst user and return a valid JWT access token."""
    from backend.models.orm import User

    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email=f"analyst_{user_id.hex[:8]}@test.com",
        hashed_password=hash_password("TestPass123"),
        full_name="Test Analyst",
        role="analyst",
    )
    db_session.add(user)
    await db_session.flush()

    return create_access_token(str(user_id), {"role": "analyst"})


@pytest_asyncio.fixture
async def admin_token(db_session: AsyncSession) -> str:
    """Register an admin user and return a valid JWT access token."""
    from backend.models.orm import User

    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email=f"admin_{user_id.hex[:8]}@test.com",
        hashed_password=hash_password("TestPass123"),
        full_name="Test Admin",
        role="admin",
    )
    db_session.add(user)
    await db_session.flush()

    return create_access_token(str(user_id), {"role": "admin"})


def auth_headers(token: str) -> dict:
    """Return Authorization header dict for a given token."""
    return {"Authorization": f"Bearer {token}"}
