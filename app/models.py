from sqlalchemy import Column, Integer, String
from .database import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    data = Column(String)
