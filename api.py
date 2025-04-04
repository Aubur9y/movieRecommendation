# api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from recommend_core import Recommender

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserProfile(BaseModel):
    gender: str     # "M" or "F"
    age: int        # 原始年龄
    occupation: int # 0~20 职业编号

class Movie(BaseModel):
    movie_id: int
    title: str
    genre: str

recommender = Recommender(
    user_model_path="data/user_tower.pt",
    item_embedding_path="data/item_embeddings.parquet",
    movies_path="data/movies_feature.parquet",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@app.post("/recommend", response_model=List[Movie])
def recommend(profile: UserProfile):
    return recommender.recommend(
        gender=profile.gender,
        age=profile.age,
        occupation=profile.occupation
    )
