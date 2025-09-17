from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from src.database.tables import UUIDTable


def convert_string_to_list(string: str):
    return list(map(float, string.split(",")))


def convert_list_to_string(float_list: list[float]):
    return ",".join(list(map(str, float_list)))


class TrainingHistory(UUIDTable):
    __tablename__ = "trainings_history"

    date_start: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    date_end: Mapped[datetime | None]
    _epoch_train_losses: Mapped[str | None]
    _epoch_validation_losses: Mapped[str | None]

    @property
    def epoch_train_losses(self):
        return (
            convert_string_to_list(self._epoch_train_losses)
            if self._epoch_train_losses is not None
            else self._epoch_train_losses
        )

    @property
    def epoch_validation_losses(self):
        return (
            convert_string_to_list(self._epoch_validation_losses)
            if self._epoch_validation_losses is not None
            else self._epoch_validation_losses
        )

    def to_dict(self) -> dict[str, UUID | datetime | list[float] | None]:
        return {
            "id": self.id,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "epoch_train_losses": self.epoch_train_losses,
            "epoch_validation_losses": self.epoch_validation_losses,
        }

    def finish(self, train_loss: list[float], validation_loss: list[float]):
        self.date_end = datetime.now(timezone.utc)
        self._epoch_train_losses = convert_list_to_string(train_loss)
        self._epoch_validation_losses = convert_list_to_string(validation_loss)


class Model(UUIDTable):
    __tablename__ = "models"

    def to_dict(self) -> dict[str, UUID]:
        return {
            "id": self.id,
        }


class Dataset(UUIDTable):
    __tablename__ = "datasets"

    def to_dict(self) -> dict[str, UUID]:
        return {
            "id": self.id,
        }
