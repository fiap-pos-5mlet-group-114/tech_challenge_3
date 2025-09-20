from uuid import UUID

from sqlalchemy import select

from src.contexts.model.tables import Model, TrainingHistory
from src.database.repositories import BaseRepo


class TrainingHistoryRepo(BaseRepo[TrainingHistory]):
    async def list_all(self):
        return list(await self.session.scalars(select(TrainingHistory)))

    async def get_ongoing(self):
        return await self.session.scalar(
            select(TrainingHistory).where(TrainingHistory.date_end.is_(None))
        )


class ModelRepo(BaseRepo[Model]):
    async def list_all(self):
        return list(await self.session.scalars(select(Model)))

    async def get_by_id(self, id_: UUID):
        return await self.session.scalar(select(Model).where(Model.id == id_))
