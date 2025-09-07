from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Message(BaseModel):
    message: str


class TrainingHistoryModel(BaseModel):
    id: UUID
    date_start: datetime
    date_end: datetime | None
    last_epoch_validation_loss: float | None
