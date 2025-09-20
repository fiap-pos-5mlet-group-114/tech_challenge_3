import sys
from asyncio import run
from pathlib import Path

from aiofiles import open as aopen
from httpx import AsyncClient
from typer import Typer

sys.path.append(Path(__file__).parents[2].as_posix())
from src.constants import MODELS_PATH

app = Typer()


async def download_model(model_slug: str = "model"):
    model_name = f"{model_slug}.pth"
    async with AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            f"https://huggingface.co/Nephilim/temperature_predictor/resolve/main/{model_name}?download=true",
        )

        response.raise_for_status()

        content = response.content

    async with aopen(MODELS_PATH / model_name, "wb") as file:
        await file.write(content)


@app.command()
def main(model_slug: str = "model"):
    run(download_model(model_slug))


if __name__ == "__main__":
    app()
