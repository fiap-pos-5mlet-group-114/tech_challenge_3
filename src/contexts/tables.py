from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from src.database.tables import UUIDTable


class TrainingHistory(UUIDTable):
    __tablename__ = "trainings_history"

    date_start: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    date_end: Mapped[datetime | None]
    last_epoch_validation_loss: Mapped[float | None]

    def to_dict(self) -> dict[str, UUID | datetime | float | None]:
        return {
            "id": self.id,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "last_epoch_validation_loss": self.last_epoch_validation_loss,
        }

    def finish(self, loss: float):
        self.date_end = datetime.now(timezone.utc)
        self.last_epoch_validation_loss = loss
