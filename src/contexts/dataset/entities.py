from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DatasetModel(BaseModel):
    id: UUID
    description: str | None


class CreateUpdateDatasetModel(BaseModel):
    description: str | None


class CreateDataModel(BaseModel):
    lat: float
    long: float
    alt: float
    hour: int
    mean_temp: float


class DataModel(BaseModel):
    id: UUID
    dataset_id: UUID
    lat: float
    long: float
    alt: float
    hour: int
    mean_temp: float


class UpdateDataModel(BaseModel):
    dataset_id: UUID | None
    lat: float | None
    long: float | None
    alt: float | None
    hour: int | None
    mean_temp: float | None
