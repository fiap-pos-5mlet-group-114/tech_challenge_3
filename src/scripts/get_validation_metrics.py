import sys
from asyncio import run
from pathlib import Path
from uuid import UUID

from rich import print as pprint
from rich.progress import track
from torch import Tensor, no_grad
from torch.utils.data import DataLoader
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from typer import Typer

sys.path.append(Path(__file__).parents[2].as_posix())
from src.constants import DEVICE, MODELS_PATH
from src.contexts.dataset import TemperatureDataset
from src.contexts.dataset.repositories import DatasetDataRepo, DatasetRepo
from src.contexts.model import TemperaturePredictor

app = Typer()


def get_metrics(
    model: TemperaturePredictor,
    dataset: TemperatureDataset,
    batch_size: int = 2048,
):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses_per_type: dict[str, float] = {}
    model.eval()
    for criterion_type in (
        MeanSquaredError,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
    ):
        loss_list = []

        criterion = criterion_type().to(DEVICE)
        with no_grad():
            for data, target in track(
                data_loader, description=f"Running for {criterion_type.__name__}"
            ):
                data, target = data.to(DEVICE), target.unsqueeze(1).to(DEVICE)

                target_pred = model(data)
                loss: Tensor = criterion(target_pred, target)

                _loss = loss.detach().item()
                loss_list.append(_loss)

        losses_per_type[criterion_type.__name__] = sum(loss_list) / len(loss_list)

    return losses_per_type


async def get_dataset_data(dataset_id: UUID):
    async with DatasetRepo() as repo:
        dataset_data_repo = DatasetDataRepo(repo.session)
        dataset_instance = await repo.get_by_id(dataset_id)
        if dataset_instance is None:
            raise ValueError(f"No dataset found with the id {dataset_id}!")

        dataset_data_list = await dataset_data_repo.get_all_by_dataset_id(
            dataset_instance.id, limit=None, offset=None
        )
    return [dataset_data.to_dict() for dataset_data in dataset_data_list]


@app.command()
def main(model_id: UUID, dataset_id: UUID):
    model = TemperaturePredictor().to(DEVICE)
    model.load(MODELS_PATH / f"{model_id}.pth")

    data = run(get_dataset_data(dataset_id))
    dataset = TemperatureDataset(data)

    metrics = get_metrics(model, dataset)

    pprint(metrics)


if __name__ == "__main__":
    app()
