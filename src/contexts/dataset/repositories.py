from uuid import UUID

from sqlalchemy import delete, func, select

from src.contexts.dataset.tables import Dataset, DatasetData
from src.database.repositories import BaseRepo


class DatasetRepo(BaseRepo[Dataset]):
    async def list_all(self):
        return list(await self.session.scalars(select(Dataset)))

    async def get_by_id(self, id_: UUID):
        return await self.session.scalar(select(Dataset).where(Dataset.id == id_))

    async def delete(self, id_: UUID):
        await self.session.execute(delete(Dataset).where(Dataset.id == id_))


class DatasetDataRepo(BaseRepo[DatasetData]):
    async def get_all_by_dataset_id(
        self, dataset_id: UUID, limit: int | None, offset: int | None
    ):
        query = select(DatasetData).where(DatasetData.dataset_id == dataset_id)
        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        return list(await self.session.scalars(query))

    async def count_all_dataset_data(self, dataset_id: UUID):
        return await self.session.scalar(
            select(func.count(DatasetData.id)).where(
                DatasetData.dataset_id == dataset_id
            )
        )

    async def get_by_id(self, id_: UUID):
        return await self.session.scalar(
            select(DatasetData).where(DatasetData.id == id_)
        )

    async def delete(self, id_: UUID):
        await self.session.execute(delete(DatasetData).where(DatasetData.id == id_))
