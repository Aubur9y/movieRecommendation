# recommend_core.py

import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.user_tower import UserTower

def discretize_age(age: int) -> int:
    if age <= 12:
        return 0
    elif age <= 18:
        return 1
    elif age <= 25:
        return 2
    elif age <= 35:
        return 3
    elif age <= 45:
        return 4
    elif age <= 60:
        return 5
    else:
        return 6

# 性别映射：M -> 0, F -> 1
def encode_gender(g: str) -> int:
    return 0 if g.upper() == "M" else 1

class Recommender:
    def __init__(self, user_model_path, item_embedding_path, movies_path, device="cpu"):
        self.device = torch.device(device)

        # 加载 user tower 模型
        self.user_tower = UserTower(num_genders=2, num_ages=7, num_occupations=21, emb_dims=32).to(self.device)
        self.user_tower.load_state_dict(torch.load(user_model_path, map_location=self.device))
        self.user_tower.eval()

        # 加载 item embedding
        self.item_df = pd.read_parquet(item_embedding_path)
        self.item_embeddings = self.item_df.drop("movieid", axis=1).values
        self.movie_ids = self.item_df["movieid"].tolist()

        # 加载电影信息
        self.movies = pd.read_parquet(movies_path).set_index("movieid")

    def recommend(self, gender: str, age: int, occupation: int, topk: int = 10):
        gender_id = encode_gender(gender)
        age_id = discretize_age(age)
        occ_id = occupation

        with torch.no_grad():
            g = torch.tensor([gender_id], dtype=torch.long).to(self.device)
            a = torch.tensor([age_id], dtype=torch.long).to(self.device)
            o = torch.tensor([occ_id], dtype=torch.long).to(self.device)

            user_vec = self.user_tower(g, a, o).cpu().numpy()  # shape: (1, 32)

        # 相似度计算
        sims = cosine_similarity(user_vec, self.item_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:topk]

        results = []
        for i in top_indices:
            mid = self.movie_ids[i]
            row = self.movies.loc[mid]
            results.append({
                "movie_id": int(mid),
                "title": row["title"],
                "genre": row["genre"]
            })

        return results
