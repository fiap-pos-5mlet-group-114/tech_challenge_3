import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from rich.logging import RichHandler

from src.constants import ASSETS_PATH, MODELS_PATH
from src.contexts.dataset.routes import router as dataset_router
from src.contexts.model.routes import router as main_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.database import create_tables

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    for path in (ASSETS_PATH, MODELS_PATH):
        path.mkdir(exist_ok=True, parents=True)

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return RedirectResponse("/docs")


app.include_router(main_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
