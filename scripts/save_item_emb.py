# scripts/save_item_emb.py

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.item_tower import ItemTower

def save_item_embeddings(
    movie_feat_path="data/movies_feature.parquet",
    model_path="data/item_tower.pt",
    output_path="data/item_embeddings.parquet",
    emb_dim=32
):
    # 加载电影特征
    movies = pd.read_parquet(movie_feat_path)
    genre_cols = [col for col in movies.columns if col.startswith("genre_")]
    movie_ids = movies["movieid"].tolist()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_tower = ItemTower(
        num_movies=max(movie_ids) + 1,
        num_years=movies["year_id"].nunique(),
        num_genres=len(genre_cols),
        emb_dim=emb_dim
    ).to(device)

    item_tower.load_state_dict(torch.load(model_path, map_location=device))
    item_tower.eval()

    movie_vectors = []
    movie_id_list = []

    with torch.no_grad():
        for i in tqdm(range(len(movies))):
            row = movies.iloc[i]
            movie_id = torch.tensor([row["movieid"]], dtype=torch.long).to(device)
            genre_vec = torch.tensor([row[genre_cols].values], dtype=torch.float32).to(device)
            year_id = torch.tensor([row["year_id"]], dtype=torch.long).to(device)

            item_vec = item_tower(movie_id, genre_vec, year_id)
            item_vec = item_vec.squeeze(0).cpu().numpy()

            movie_vectors.append(item_vec)
            movie_id_list.append(row["movieid"])

    # 保存为 parquet
    df = pd.DataFrame(movie_vectors)
    df["movieid"] = movie_id_list

    if os.path.exists(output_path):
        os.remove(output_path)

    df.to_parquet(output_path, index=False)
    print(f"✅ 已保存 item embedding 到 {output_path}")


if __name__ == "__main__":
    save_item_embeddings()
