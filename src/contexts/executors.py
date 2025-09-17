from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from uuid import uuid4

from polars import DataFrame, read_csv
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader, Subset, random_split

from src.constants import DATASETS_PATH, MODELS_PATH
from src.contexts.dataset import TemperatureDataset
from src.contexts.entities import TrainingParams
from src.contexts.model import TemperaturePredictor
from src.contexts.repositories import DatasetRepo, ModelRepo, TrainingHistoryRepo
from src.contexts.tables import Model


def create_train_validation_datasets(dataframe: DataFrame):
    dados = TemperatureDataset(dataframe)
    return random_split(dados, [0.8, 0.2])


def train(
    model: TemperaturePredictor,
    train_dataset: Subset,
    validation_dataset: Subset,
    save_path: Path,
    logger: Logger,
    epochs: int = 20,
    batch_size: int = 2048,
):
    criterion = nn.MSELoss().cuda()
    optimizer = optim.AdamW(model.parameters())
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    epoch_train_losses: list[float] = []
    epoch_validation_losses: list[float] = []

    for e in range(epochs):
        model = model.train()

        train_losses: list[float] = []
        validation_losses: list[float] = []

        for data, target in train_data_loader:
            data, target = data.cuda(), target.unsqueeze(1).cuda()

            optimizer.zero_grad()
            target_pred = model(data)
            loss = criterion(target_pred, target)

            loss.backward()
            optimizer.step()

            _loss = loss.detach().item()
            train_losses.append(_loss)

        model.eval()
        with no_grad():
            for data, target in validation_data_loader:
                data, target = data.cuda(), target.unsqueeze(1).cuda()

                target_pred = model(data)
                loss = criterion(target_pred, target)

                _loss = loss.detach().item()
                validation_losses.append(_loss)

        epoch_train_losses.append(sum(train_losses) / len(train_losses))
        epoch_validation_losses.append(sum(validation_losses) / len(validation_losses))
        logger.info(
            f"Epoch {e + 1} train loss: {epoch_train_losses[-1]:.04f}; validation loss: {epoch_validation_losses[-1]:.04f}"
        )

        model.save(save_path)

    return epoch_train_losses, epoch_validation_losses


async def train_model(logger: Logger, training_params: TrainingParams):
    async with TrainingHistoryRepo() as training_history_repo:
        history = await training_history_repo.get_ongoing()
        if history is None:
            return

        dataset_repo = DatasetRepo(training_history_repo.session)
        dataset_instance = await dataset_repo.get_by_id(training_params.dataset_id)
        if dataset_instance is None:
            raise ValueError(
                f"No dataset found with the id {training_params.dataset_id}!"
            )

        dataframe = read_csv(DATASETS_PATH / f"{dataset_instance.id}.csv")
        train_dataset, validation_dataset = create_train_validation_datasets(dataframe)
        model = TemperaturePredictor().cuda()

        model_repo = ModelRepo(training_history_repo.session)
        if training_params.model_id is not None:
            model_instance = await model_repo.get_by_id(training_params.model_id)
            if model_instance:
                model.load(MODELS_PATH / f"{model_instance.id}.pth")

        model_instance = Model(id=uuid4())

        start = datetime.now(timezone.utc)
        logger.info(
            f"Starting training at [green]{start.strftime('%d/%m/%Y, %H:%M:%S')}[/]",
            extra={"markup": True},
        )
        train_loss, validation_loss = train(
            model,
            train_dataset,
            validation_dataset,
            MODELS_PATH / f"{model_instance.id}.pth",
            logger,
            training_params.epochs,
            training_params.batch_size,
        )
        end = datetime.now(timezone.utc)
        logger.info(
            f"Training finished at [green]{end.strftime('%d/%m/%Y, %H:%M:%S')}[/]",
            extra={"markup": True},
        )
        history.finish(train_loss, validation_loss)

        training_history_repo.add(history)
        model_repo.add(model_instance)

        await training_history_repo.commit()
