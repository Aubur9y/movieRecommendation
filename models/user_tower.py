import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_genders, num_ages, num_occupations, emb_dims=32):
        super(UserTower, self).__init__()

        self.gender_emb = nn.Embedding(num_genders, emb_dims)
        self.age_emb = nn.Embedding(num_ages, emb_dims)
        self.occupation_emb = nn.Embedding(num_occupations, emb_dims)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dims * 3, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dims)
        )
    
    def forward(self, gender, age, occupation):
        gender_vec = self.gender_emb(gender)
        age_vec = self.age_emb(age)
        occup_vec = self.occupation_emb(occupation)

        x = torch.concat([gender_vec, age_vec, occup_vec], dim=-1)
        out = self.mlp(x)
        out = F.normalize(out, dim=-1)
        return out