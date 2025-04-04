# scripts/feature.py

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def split_genres(df):
    all_genres = set()
    for g in df["genre"]:
        all_genres.update(g.split("|"))

    all_genres = sorted(list(all_genres))
    genre2id = {g: i for i, g in enumerate(all_genres)}

    genre_matrix = np.zeros((len(df), len(all_genres)))
    for idx, g_list in enumerate(df["genre"]):
        for g in g_list.split("|"):
            genre_matrix[idx][genre2id[g]] = 1

    genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in all_genres])
    return pd.concat([df.reset_index(drop=True), genre_df], axis=1), all_genres

def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else 0

def build_movie_features(movie_path="data/movies.parquet", out_path="data/movies_feature.parquet"):
    movies = pd.read_parquet(movie_path)

    # genre -> one-hot
    movies, genre_list = split_genres(movies)

    # title -> year
    movies["year"] = movies["title"].apply(extract_year)

    # year -> label encode
    le_year = LabelEncoder()
    movies["year_id"] = le_year.fit_transform(movies["year"])

    joblib.dump(le_year, "data/label_year.pkl")  # 保存编码器（可选）

    if os.path.exists(out_path):
        os.remove(out_path)

    movies.to_parquet(out_path, index=False)

    print(f"✅ 保存电影特征到 {out_path}")
    print(f"🎭 Genre 维度: {len(genre_list)}")
    print(f"📆 年份范围: {movies['year'].min()} ~ {movies['year'].max()}")

if __name__ == "__main__":
    build_movie_features()
