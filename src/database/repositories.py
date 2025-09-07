from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from src.database import async_session


class BaseRepo[T]:
    session: AsyncSession

    async def __aenter__(self):
        a_session = async_session()
        self.session = a_session
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    def add(self, instance: T):
        self.session.add(instance)

    def add_all(self, instances: Sequence[T]):
        self.session.add_all(instances)

    async def commit(self):
        await self.session.commit()
