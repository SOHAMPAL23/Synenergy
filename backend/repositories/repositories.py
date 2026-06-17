"""
EnerVision AI - Repository Pattern
Generic async CRUD base + domain-specific repositories.
"""

import uuid
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import Base

ModelT = TypeVar("ModelT", bound=Base)


# ─── Generic Base Repository ──────────────────────────────────────────────────

class BaseRepository(Generic[ModelT]):
    """Async CRUD operations for any SQLAlchemy model."""

    def __init__(self, model: Type[ModelT], db: AsyncSession) -> None:
        self._model = model
        self._db = db

    async def get(self, id_: uuid.UUID) -> Optional[ModelT]:
        return await self._db.get(self._model, id_)

    async def get_all(
        self, skip: int = 0, limit: int = 100, **filters: Any
    ) -> List[ModelT]:
        stmt = select(self._model)
        for attr, val in filters.items():
            stmt = stmt.where(getattr(self._model, attr) == val)
        stmt = stmt.offset(skip).limit(limit)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def create(self, obj: ModelT) -> ModelT:
        self._db.add(obj)
        await self._db.flush()
        await self._db.refresh(obj)
        return obj

    async def update(self, id_: uuid.UUID, data: Dict[str, Any]) -> Optional[ModelT]:
        obj = await self.get(id_)
        if not obj:
            return None
        for k, v in data.items():
            setattr(obj, k, v)
        await self._db.flush()
        await self._db.refresh(obj)
        return obj

    async def delete(self, id_: uuid.UUID) -> bool:
        obj = await self.get(id_)
        if not obj:
            return False
        await self._db.delete(obj)
        await self._db.flush()
        return True

    async def count(self, **filters: Any) -> int:
        stmt = select(func.count()).select_from(self._model)
        for attr, val in filters.items():
            stmt = stmt.where(getattr(self._model, attr) == val)
        result = await self._db.execute(stmt)
        return result.scalar_one()


# ─── User Repository ──────────────────────────────────────────────────────────

class UserRepository(BaseRepository):
    from backend.models.orm import User

    def __init__(self, db: AsyncSession) -> None:
        from backend.models.orm import User
        super().__init__(User, db)

    async def get_by_email(self, email: str):
        from backend.models.orm import User
        stmt = select(User).where(User.email == email)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_users(self, skip: int = 0, limit: int = 100):
        from backend.models.orm import User
        stmt = select(User).where(User.is_active == True).offset(skip).limit(limit)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())


# ─── Energy Record Repository ─────────────────────────────────────────────────

class EnergyRecordRepository(BaseRepository):

    def __init__(self, db: AsyncSession) -> None:
        from backend.models.orm import EnergyRecord
        super().__init__(EnergyRecord, db)

    async def get_by_user(self, user_id: uuid.UUID, limit: int = 10000):
        from backend.models.orm import EnergyRecord
        stmt = (
            select(EnergyRecord)
            .where(EnergyRecord.user_id == user_id)
            .order_by(EnergyRecord.timestamp.asc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_user(self, user_id: uuid.UUID) -> int:
        from backend.models.orm import EnergyRecord
        stmt = delete(EnergyRecord).where(EnergyRecord.user_id == user_id)
        result = await self._db.execute(stmt)
        return result.rowcount


# ─── Forecast Repository ──────────────────────────────────────────────────────

class ForecastRepository(BaseRepository):

    def __init__(self, db: AsyncSession) -> None:
        from backend.models.orm import Forecast
        super().__init__(Forecast, db)

    async def get_latest_by_user(self, user_id: uuid.UUID):
        from backend.models.orm import Forecast
        stmt = (
            select(Forecast)
            .where(Forecast.user_id == user_id, Forecast.is_latest == True)
            .order_by(Forecast.created_at.desc())
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def mark_all_old(self, user_id: uuid.UUID) -> None:
        from backend.models.orm import Forecast
        stmt = (
            update(Forecast)
            .where(Forecast.user_id == user_id)
            .values(is_latest=False)
        )
        await self._db.execute(stmt)



# ─── Recommendation Repository ────────────────────────────────────────────────

class RecommendationRepository(BaseRepository):

    def __init__(self, db: AsyncSession) -> None:
        from backend.models.orm import Recommendation
        super().__init__(Recommendation, db)

    async def get_active_by_user(self, user_id: uuid.UUID):
        from backend.models.orm import Recommendation
        stmt = (
            select(Recommendation)
            .where(Recommendation.user_id == user_id, Recommendation.is_active == True)
            .order_by(Recommendation.created_at.desc())
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def deactivate_all_for_user(self, user_id: uuid.UUID) -> None:
        from backend.models.orm import Recommendation
        stmt = (
            update(Recommendation)
            .where(Recommendation.user_id == user_id)
            .values(is_active=False)
        )
        await self._db.execute(stmt)
