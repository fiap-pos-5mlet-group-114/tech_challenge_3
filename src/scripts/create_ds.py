import sys
from asyncio import run
from pathlib import Path
from uuid import uuid4

from polars import read_csv
from rich.progress import track
from typer import Typer

sys.path.append(Path(__file__).parents[2].as_posix())
from src.contexts.dataset.repositories import DatasetDataRepo, DatasetRepo
from src.contexts.dataset.tables import Dataset, DatasetData
from src.database import create_tables

BASE_PATH = Path(__file__).parents[2] / "data"

OUT_PATH = BASE_PATH / "out"
app = Typer()


async def create_ds_with_data(year: int):
    await create_tables()
    params = read_csv(OUT_PATH / f"{year}.csv")
    dataset = Dataset(id=uuid4())
    dataset_data_list: list[DatasetData] = []
    for row in track(params.rows(named=True)):
        dataset_data_list.append(
            DatasetData(
                id=uuid4(),
                dataset_id=dataset.id,
                lat=row["LATITUDE"],
                long=row["LONGITUDE"],
                alt=row["ALTITUDE"],
                hour=row["Hora"],
                month=row["Mes"],
                day=row["Dia"],
                mean_temp=row["temp_mean"],
            )
        )

    async with DatasetRepo() as repo:
        dataset_data_repo = DatasetDataRepo(repo.session)

        repo.add(dataset)
        dataset_data_repo.add_all(dataset_data_list)

        await repo.commit()


@app.command()
def main(year: int):
    run(create_ds_with_data(year))


if __name__ == "__main__":
    app()
