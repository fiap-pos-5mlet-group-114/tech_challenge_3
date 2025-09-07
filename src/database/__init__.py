from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine("sqlite+aiosqlite:///database.db")
async_session = async_sessionmaker(engine, autoflush=True)


async def create_tables():
    from src.contexts.tables import TrainingHistory
    from src.database.tables import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
