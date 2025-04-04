import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, num_movies, num_years, num_genres, emb_dim=32):
        super().__init__()
        self.movie_emb = nn.Embedding(num_movies, emb_dim)
        self.year_emb = nn.Embedding(num_years, emb_dim)
        self.genre_emb = nn.Linear(num_genres, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, movie_id, genre_vec, year_id):
        movie_vec = self.movie_emb(movie_id)
        year_vec = self.year_emb(year_id)
        genre_vec = self.genre_emb(genre_vec.float())

        x = torch.concat([movie_vec, year_vec, genre_vec], dim=-1)
        out = self.mlp(x)

        return F.normalize(out, dim=-1)