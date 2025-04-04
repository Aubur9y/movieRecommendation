from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets.user_item_dataset import UserItemDataset
from models.user_tower import UserTower
from models.item_tower import ItemTower
import pandas as pd
import numpy as np

EMD_DIM = 32
BATCH_SIZE = 512
EPOCHS = 2
LR = 1e-3

df = pd.read_parquet("data/train_user_tower.parquet")
movie_df = pd.read_parquet("data/movies_feature.parquet")

genre_cols = [col for col in movie_df.columns if col.startswith("genre_")]
movie_df = movie_df.set_index("movieid")

dataset = UserItemDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_tower = UserTower(
    num_genders=df["gender_id"].max() + 1,
    num_ages=df["age_id"].max() + 1,
    num_occupations=df["occupation_id"].max() + 1,
    emb_dims=EMD_DIM
).to(device)

item_tower = ItemTower(
    num_movies=movie_df.index.max() + 1,
    num_years=movie_df["year_id"].nunique(),
    num_genres=len(genre_cols),
    emb_dim=EMD_DIM,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    list(user_tower.parameters()) + list(item_tower.parameters()), lr=LR
)

for epoch in range(EPOCHS):
    user_tower.train()
    item_tower.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        gender = batch["gender_id"].to(device)
        age = batch["age_id"].to(device)
        occupation = batch["occupation_id"].to(device)
        movie_ids = batch["movie_id"].to(device)
        label = batch["label"].to(device)

        genre_vecs = []
        year_ids = []

        for mid in movie_ids.cpu().numpy():
            row = movie_df.loc[mid]
            genre_vecs.append(row[genre_cols].values)
            year_ids.append(row["year_id"])

        genre_vecs = torch.from_numpy(np.array(genre_vecs, dtype=np.float32)).to(device)
        year_ids = torch.tensor(year_ids, dtype=torch.long).to(device)

        user_vec = user_tower(gender, age, occupation)
        item_vec = item_tower(movie_ids, genre_vecs, year_ids)

        logits = torch.sum(user_vec * item_vec, dim=1) # cosine similarity, since we already normalised both vectors, the denominator is 1, so its basically the dot product of both vectors
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

torch.save(user_tower.state_dict(), "user_tower.pt")
torch.save(item_tower.state_dict(), "item_tower.pt")
print("Model Saved!")