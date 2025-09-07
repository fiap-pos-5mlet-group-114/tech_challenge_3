from sqlalchemy import select

from src.contexts.tables import TrainingHistory
from src.database.repositories import BaseRepo


class TrainingHistoryRepo(BaseRepo[TrainingHistory]):
    async def list_all(self):
        return list(await self.session.scalars(select(TrainingHistory)))

    async def get_ongoing(self):
        return await self.session.scalar(
            select(TrainingHistory).where(TrainingHistory.date_end.is_(None))
        )
