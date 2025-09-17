from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Message(BaseModel):
    message: str


class TrainingHistoryModel(BaseModel):
    id: UUID
    date_start: datetime
    date_end: datetime | None
    epoch_train_losses: list[float] | None
    epoch_validation_losses: list[float] | None


class TrainingParams(BaseModel):
    dataset_id: UUID
    model_id: UUID | None
    epochs: int = 20
    batch_size: int = 2048


class PredictParams(BaseModel):
    lat: float
    long: float
    alt: float
    hour: int


class Predict(BaseModel):
    model_id: UUID
    params: list[PredictParams]
