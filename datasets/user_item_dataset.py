from torch.utils.data import Dataset
import torch

class UserItemDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            sample = {
                "gender_id": torch.tensor(row["gender_id"], dtype=torch.long),
                "age_id": torch.tensor(row["age_id"], dtype=torch.long),
                "occupation_id": torch.tensor(row["occupation_id"], dtype=torch.long),
                "movie_id": torch.tensor(row["movieid"], dtype=torch.long),
                "label": torch.tensor(row["label"], dtype=torch.float)
            }
            return sample
        except Exception as e:
            print(f"❌ Error at index {index}")
            print(row)
            raise e

class MovieFeatureDataset(Dataset):
    def __init__(self, movie_df):
        self.movie_df = movie_df.reset_index(drop=True)
        self.genre_cols = [col for col in movie_df.columns if col.startswith("genre_")]

    def __len__(self):
        return len(self.movie_df)

    def __getitem__(self, idx):
        row = self.movie_df.iloc[idx]

        sample = {
            "movie_id": torch.tensor(int(row["movieid"]), dtype=torch.long), 
            "genre_vec": torch.tensor(row[self.genre_cols].values, dtype=torch.float),
            "year_id": torch.tensor(int(row["year_id"]), dtype=torch.long)
        }
        return sample