from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class RecommendationDataset(Dataset):
    def __init__(self, data_path, features, contexts, dataset_type = 'train', read_type='csv'):
        if read_type=='csv':
            full_df = pd.read_csv(data_path)
        elif read_type=='pkl':
            full_df = pd.read_pickle(data_path.replace('csv','pkl'))
        self.all_users = list(full_df['user'].unique())
        self.all_movies = list(full_df['product'].unique())

        X_train, X_test, y_train, y_test = train_test_split(full_df.loc[:, full_df.columns != 'y'], 
                                                            full_df['y'], 
                                                            test_size=.2, 
                                                            random_state=42)
        del full_df

        if dataset_type == 'train':
            tmp_x_user = X_train['user'].to_numpy()
            tmp_x_movie = X_train['product'].to_numpy()
            tmp_x_feats = X_train[features].to_numpy()
            tmp_x_contexts = X_train[contexts].to_numpy()
            tmp_y = y_train.to_numpy()
        elif dataset_type == 'test':
            tmp_x_user = X_test['user'].to_numpy()
            tmp_x_movie = X_test['product'].to_numpy()
            tmp_x_feats = X_test[features].to_numpy()
            tmp_x_contexts = X_test[contexts].to_numpy()
            tmp_y = y_test.to_numpy()

        self.x_user = torch.tensor(tmp_x_user, dtype=torch.int64)
        self.x_movie = torch.tensor(tmp_x_movie, dtype=torch.int64)
        self.x_feats_tensors = torch.tensor(tmp_x_feats, dtype=torch.int64)
        self.x_contexts_tensors = torch.tensor(tmp_x_contexts, dtype=torch.int64)
        self.y_data = torch.tensor(tmp_y, dtype=torch.int64).reshape(-1, 1)


    def __len__(self):
        return len(self.x_user)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # users_vector = torch.tensor([1 if x==self.x_user[idx] else 0 for x in self.all_users], 
        #                             dtype=torch.int64)
        # movies_vector = torch.tensor([1 if x==self.x_movie[idx] else 0 for x in self.all_movies], 
        #                             dtype=torch.int64)
        users_vector = self.x_user[idx]
        movies_vector = self.x_movie[idx]
        feats_vector = self.x_feats_tensors[idx]
        context_vector = self.x_contexts_tensors[idx]
        y_vector = self.y_data[idx]

        return {'user': users_vector, 
                'movie': movies_vector, 
                'movie_features': feats_vector,
                'user_context': context_vector}, y_vector