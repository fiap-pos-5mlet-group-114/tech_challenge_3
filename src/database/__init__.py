from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.constants import MODELS_PATH
from src.utils import is_uuid

engine = create_async_engine("sqlite+aiosqlite:///database.db")
async_session = async_sessionmaker(engine, autoflush=True)


def create_models_from_files(session: AsyncSession):
    from src.contexts.tables import Model

    models: list[Model] = []
    for file in MODELS_PATH.iterdir():
        file_name = file.stem
        if is_uuid(file_name):
            continue
        file_uuid = uuid4()
        instance = Model(id=file_uuid)
        models.append(instance)
        file.rename(file.parent / f"{instance.id}.pth")

    session.add_all(models)


async def create_tables():
    from src.contexts.dataset.tables import Dataset, DatasetData
    from src.contexts.tables import Model, TrainingHistory
    from src.database.tables import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        create_models_from_files(session)
        await session.commit()
