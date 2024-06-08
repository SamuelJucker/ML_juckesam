from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, database

router = APIRouter()

@router.post("/upload")
def upload_dataset(name: str, data: str, db: Session = Depends(database.get_db)):
    db_dataset = models.Dataset(name=name, data=data)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

@router.get("/datasets")
def get_datasets(skip: int = 0, limit: int = 10, db: Session = Depends(database.get_db)):
    datasets = db.query(models.Dataset).offset(skip).limit(limit).all()
    return datasets
