from fastapi import FastAPI
from .database import engine, Base
from .routers import router as dataset_router

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(dataset_router, prefix="/api")
