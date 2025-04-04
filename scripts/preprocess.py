import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import os
import joblib

ratings = pd.read_parquet("data/ratings.parquet")
users = pd.read_parquet("data/users.parquet")

positive = ratings[ratings['rating'] >= 4].copy() # keep ratings >= 4 as positive
positive = positive.merge(users, left_on='userid', right_on='userId')

positive = positive.dropna(subset=["gender", "age", "occupation"])

le_gender = LabelEncoder()

le_occ = LabelEncoder()

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

positive['gender_id'] = le_gender.fit_transform(positive["gender"])
positive['age_id'] = positive['age'].apply(discretize_age)
positive['occupation_id'] = le_occ.fit_transform(positive['occupation'])

positive_samples = positive[["userid", "movieid", "gender_id", "age_id", "occupation_id"]].copy()
positive_samples["label"] = 1

def generate_negative_samples(positive_df, all_movie_ids, neg_per_user=3):
    data = []
    grouped = positive_df.groupby("userid")

    for user_id, group in grouped:
        pos_movies = set(group["movieid"])
        candidate_neg = list(all_movie_ids - pos_movies)
        if not candidate_neg:
            continue

        neg_movies = random.sample(candidate_neg, min(len(candidate_neg), neg_per_user))
        user_info = group.iloc[0][['gender_id', 'age_id', 'occupation_id']]

        for movie_id in neg_movies:
            data.append({
                "userid": user_id,
                "movieid": movie_id,
                "gender_id": user_info["gender_id"],
                "age_id": user_info["age_id"],
                "occupation_id": user_info["occupation_id"],
                "label": 0
            })

    return pd.DataFrame(data)
    
all_movie_ids = set(ratings["movieid"].unique())
negative_samples = generate_negative_samples(positive_samples, all_movie_ids)

train_df = pd.concat([positive_samples, negative_samples], ignore_index=True)

train_df = train_df.dropna(subset=["gender_id", "age_id", "occupation_id", "movieid", "label"])

# Ensure the file is not locked or in use by another process
output_file = 'data/train_user_tower.parquet'
if os.path.exists(output_file):
    os.remove(output_file)  # Remove the file if it already exists

train_df.to_parquet(output_file, index=False)

print(f"训练样本数量：{len(train_df)}, 正样本数量：{len(positive_samples)}, 负样本数量：{len(negative_samples)}")