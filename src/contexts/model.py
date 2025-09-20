from pathlib import Path

from torch import Tensor, load, nn, save


class TemperaturePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def save(self, path: Path):
        save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(load(path, weights_only=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
