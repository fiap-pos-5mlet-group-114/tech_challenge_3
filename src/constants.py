from pathlib import Path

from torch.cuda import is_available

BASE_PATH = Path(__file__).parents[1]
ASSETS_PATH = BASE_PATH / "assets"
MODELS_PATH = ASSETS_PATH / "models"
DEVICE = "cuda" if is_available() else "cpu"
