from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.contexts.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.database import create_tables

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return RedirectResponse("/docs")


app.include_router(router, prefix="/api")
