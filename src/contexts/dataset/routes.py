from typing import Annotated
from uuid import UUID

from fastapi import Query
from fastapi.routing import APIRouter

from src.contexts.dataset.entities import (
    CreateDataModel,
    CreateUpdateDatasetModel,
    DataModel,
    DatasetModel,
    UpdateDataModel,
)
from src.contexts.dataset.repositories import DatasetDataRepo, DatasetRepo
from src.contexts.dataset.tables import Dataset, DatasetData

router = APIRouter(prefix="/datasets", tags=["Dataset"])


@router.get("", response_model=list[DatasetModel])
async def get_all():
    async with DatasetRepo() as repo:
        datasets = await repo.list_all()

    return [dataset.to_dict() for dataset in datasets]


@router.post("", status_code=201, response_model=DatasetModel)
async def create(params: CreateUpdateDatasetModel):
    dataset = Dataset(description=params.description)
    async with DatasetRepo() as repo:
        repo.add(dataset)
        await repo.commit()
        await repo.session.refresh(dataset)

    return dataset.to_dict()


@router.patch("/{id}", response_model=DatasetModel)
async def update(id: UUID, params: CreateUpdateDatasetModel):
    async with DatasetRepo() as repo:
        dataset = await repo.get_by_id(id)
        if dataset is None:
            raise ValueError(f"Dataset with id {id} not found!")
        dataset.description = params.description
        repo.add(dataset)
        await repo.commit()
        await repo.session.refresh(dataset)

    return dataset.to_dict()


@router.delete("/{id}", status_code=204)
async def delete(id: UUID):
    async with DatasetRepo() as repo:
        await repo.delete(id)
        await repo.commit()


@router.get("/{dataset_id}/data", response_model=list[DataModel])
async def get_dataset_data(
    dataset_id: UUID,
    limit: Annotated[int, Query(le=1000.0, gt=0.0)] = 100,
    offset: int = 0,
):
    async with DatasetDataRepo() as repo:
        dataset_data_list = await repo.get_all_by_dataset_id(dataset_id, limit, offset)

    return [dataset_data.to_dict() for dataset_data in dataset_data_list]


@router.post("/{dataset_id}/data", status_code=201, response_model=DataModel)
async def add_data(dataset_id: UUID, params: CreateDataModel):
    dataset_data = DatasetData(
        dataset_id=dataset_id,
        lat=params.lat,
        long=params.long,
        alt=params.alt,
        hour=params.hour,
        month=params.month,
        day=params.day,
        mean_temp=params.mean_temp,
    )
    async with DatasetDataRepo() as repo:
        repo.add(dataset_data)
        await repo.commit()
        await repo.session.refresh(dataset_data)

    return dataset_data.to_dict()


@router.patch("/{dataset_id}/data/{data_id}", response_model=DataModel)
async def update_data(dataset_id: UUID, data_id: UUID, params: UpdateDataModel):
    async with DatasetDataRepo() as repo:
        dataset_data = await repo.get_by_id(data_id)
        if dataset_data is None:
            raise ValueError(f"Dataset data with id {data_id} not found!")

        for key, value in params.model_dump().items():
            if value is None:
                continue
            setattr(dataset_data, key, value)
        repo.add(dataset_data)
        await repo.commit()
        await repo.session.refresh(dataset_data)

    return dataset_data.to_dict()


@router.delete("/{dataset_id}/data/{data_id}", status_code=204)
async def delete_data(dataset_id: UUID, data_id: UUID):
    async with DatasetDataRepo() as repo:
        await repo.delete(data_id)
        await repo.commit()
