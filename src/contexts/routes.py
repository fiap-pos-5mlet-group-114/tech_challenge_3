from asyncio import get_event_loop

from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

from src.contexts.entities import Message, TrainingHistoryModel
from src.contexts.executors import train_model
from src.contexts.repositories import TrainingHistoryRepo
from src.contexts.tables import TrainingHistory

router = APIRouter()


@router.get("/collect-data")
async def collect_data():
    pass


@router.post(
    "/train",
    response_model=TrainingHistoryModel,
    responses={406: {"model": Message}},
    status_code=201,
)
async def train():
    async with TrainingHistoryRepo() as repo:
        if await repo.get_ongoing() is not None:
            return JSONResponse(
                status_code=406, content={"message": "Model already training!"}
            )
        history = TrainingHistory()
        repo.add(history)
        await repo.commit()
        await repo.session.refresh(history)

    get_event_loop().create_task(train_model())
    return history.to_dict()


@router.get("/training-history", response_model=list[TrainingHistoryModel])
async def training_history():
    async with TrainingHistoryRepo() as repo:
        trainings_history = await repo.list_all()
    return list(map(lambda t_h: t_h.to_dict(), trainings_history))


@router.post("/predict")
async def predict():
    pass
