from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.constants import DATASETS_PATH, MODELS_PATH
from src.utils import is_uuid

engine = create_async_engine("sqlite+aiosqlite:///database.db")
async_session = async_sessionmaker(engine, autoflush=True)


def create_models_and_datasets_from_files(session: AsyncSession):
    from src.contexts.tables import Dataset, Model

    models: list[Model] = []
    datasets: list[Dataset] = []
    for path, model_class, instance_list, file_ext in zip(
        (MODELS_PATH, DATASETS_PATH),
        (Model, Dataset),
        (models, datasets),
        ("pth", "csv"),
    ):
        for file in path.iterdir():
            file_name = file.stem
            if is_uuid(file_name):
                continue
            file_uuid = uuid4()
            instance = model_class(id=file_uuid)
            instance_list.append(instance)  # type: ignore
            file.rename(file.parent / f"{instance.id}.{file_ext}")

    session.add_all((*models, *datasets))


async def create_tables():
    from src.contexts.tables import Dataset, Model, TrainingHistory
    from src.database.tables import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        create_models_and_datasets_from_files(session)
        await session.commit()
