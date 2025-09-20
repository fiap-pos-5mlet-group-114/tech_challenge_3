from uuid import UUID

from torch import Tensor
from torch.utils.data import Dataset


class TemperatureDataset(Dataset):
    def __init__(self, data: list[dict[str, UUID | int | float]]) -> None:
        super().__init__()
        data_rows: list[tuple[int | float]] = [
            (
                row["lat"],
                row["long"],
                row["alt"],
                row["hour"],
                row["month"],
                row["day"],
                row["mean_temp"],
            )
            for row in data
        ]  # type: ignore
        tensor = Tensor(data_rows)
        self.data = tensor[:, :-1]
        self.target = tensor[:, -1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index], self.target[index]
