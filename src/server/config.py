import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from rich.logging import RichHandler

from src.constants import ASSETS_PATH, DATASETS_PATH, MODELS_PATH
from src.contexts.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.database import create_tables

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    for path in (ASSETS_PATH, MODELS_PATH, DATASETS_PATH):
        path.mkdir(exist_ok=True, parents=True)

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return RedirectResponse("/docs")


app.include_router(router, prefix="/api")
