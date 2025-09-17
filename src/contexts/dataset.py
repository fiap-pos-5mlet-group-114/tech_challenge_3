from polars import DataFrame
from torch import Tensor
from torch.utils.data import Dataset


class TemperatureDataset(Dataset):
    def __init__(self, csv: DataFrame) -> None:
        super().__init__()
        temp = []
        for linha in csv.rows():
            temp.append(linha)
        data = Tensor(temp)
        self.data = data[:, -5:-1]
        self.target = data[:, -1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index], self.target[index]
