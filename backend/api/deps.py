"""
EnerVision AI - FastAPI Route Dependencies
JWT authentication, RBAC, and shared dependencies.
"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import decode_access_token, decode_refresh_token
from backend.database.session import get_db
from backend.models.orm import User

bearer_scheme = HTTPBearer(auto_error=False)


# ─── JWT Token Extraction ─────────────────────────────────────────────────────

async def get_current_user_id(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)
    ] = None,
) -> str:
    """Extract and validate JWT access token; return user_id (str UUID)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide Bearer token.",
        )
    user_id = decode_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    return user_id


async def get_current_user(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve user_id → User ORM object, check active status."""
    import uuid
    from backend.repositories import UserRepository
    repo = UserRepository(db)
    user = await repo.get(uuid.UUID(user_id))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )
    return user


# ─── RBAC Helpers ─────────────────────────────────────────────────────────────

def require_roles(*roles: str):
    """Dependency factory: require user to have one of the given roles."""
    async def _check(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' does not have access. "
                       f"Required: {roles}",
            )
        return current_user
    return _check


def require_admin():
    return require_roles("admin")


def require_analyst_or_above():
    return require_roles("admin", "analyst")
