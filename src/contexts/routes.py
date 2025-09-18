from asyncio import get_event_loop
from logging import getLogger

from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from torch import Tensor

from src.constants import MODELS_PATH
from src.contexts.dataset.repositories import DatasetDataRepo
from src.contexts.entities import Message, Predict, TrainingHistoryModel, TrainingParams
from src.contexts.executors import train_model
from src.contexts.model import TemperaturePredictor
from src.contexts.repositories import TrainingHistoryRepo
from src.contexts.tables import TrainingHistory

router = APIRouter(tags=["Training"])


@router.get("/collect-data")
async def collect_data():
    pass


@router.post(
    "/train",
    response_model=TrainingHistoryModel,
    responses={406: {"model": Message}},
    status_code=201,
)
async def train(params: TrainingParams):
    async with TrainingHistoryRepo() as repo:
        if await repo.get_ongoing() is not None:
            return JSONResponse(
                status_code=406, content={"message": "Model already training!"}
            )
        dataset_data_repo = DatasetDataRepo(repo.session)
        amount_of_data = await dataset_data_repo.count_all_dataset_data(
            params.dataset_id
        )
        if amount_of_data is not None and amount_of_data < 5:
            return JSONResponse(
                status_code=406,
                content={
                    "message": f"Not enough samples in the dataset! {amount_of_data} out of 5 required!"
                },
            )
        history = TrainingHistory()
        repo.add(history)
        await repo.commit()
        await repo.session.refresh(history)

    logging_msg = (
        f'the model with id [red]"{params.model_id}"[/]'
        if params.model_id
        else "a new model"
    )
    logger = getLogger("training")
    logger.info(
        f'Invoking training of {logging_msg} using the dataset of id [red]"{params.dataset_id}"[/]',
        extra={"markup": True},
    )
    get_event_loop().create_task(train_model(logger, params))
    return history.to_dict()


@router.get("/training-history", response_model=list[TrainingHistoryModel])
async def training_history():
    async with TrainingHistoryRepo() as repo:
        trainings_history = await repo.list_all()
    return list(map(lambda t_h: t_h.to_dict(), trainings_history))


@router.post("/predict")
async def predict(predict_params: Predict):
    model = TemperaturePredictor().cuda().eval()
    model.load(MODELS_PATH / f"{predict_params.model_id}.pth")

    pred: Tensor = model(
        Tensor(
            [
                [param.lat, param.long, param.alt, param.hour]
                for param in predict_params.params
            ]
        )
        .unsqueeze(0)
        .cuda()
    )
    return {"mean_temp": pred.squeeze(0).detach().item()}
