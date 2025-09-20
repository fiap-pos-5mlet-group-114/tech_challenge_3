from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


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
    epochs: int = Field(default=20, ge=1, le=100)
    batch_size: int = Field(default=2048, ge=1, le=4096)


class PredictParams(BaseModel):
    lat: float
    long: float
    alt: float
    hour: int
    month: int
    day: int


class Predict(BaseModel):
    model_id: UUID
    params: list[PredictParams]
