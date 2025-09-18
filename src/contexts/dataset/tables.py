from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.tables import UUIDTable


class Dataset(UUIDTable):
    __tablename__ = "datasets"
    description: Mapped[str | None]
    data: Mapped[list["DatasetData"]] = relationship(cascade="all,delete")

    def to_dict(self) -> dict[str, UUID | str | None]:
        return {"id": self.id, "description": self.description}


class DatasetData(UUIDTable):
    __tablename__ = "datasets_data"

    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("datasets.id"), index=True)

    lat: Mapped[float]
    long: Mapped[float]
    alt: Mapped[float]
    hour: Mapped[int]
    mean_temp: Mapped[float]

    def to_dict(self) -> dict[str, UUID | int | float]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "lat": self.lat,
            "long": self.long,
            "alt": self.alt,
            "hour": self.hour,
            "mean_temp": self.mean_temp,
        }
