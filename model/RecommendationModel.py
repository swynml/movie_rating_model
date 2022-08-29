import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class RecommendationModel(pl.LightningModule):
    def __init__(self, lr, users_count, movies_count, embedding_size, features_count, context_count):
        super(RecommendationModel, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

        self.user_embedding = nn.Embedding(users_count, embedding_size)
        self.movie_embedding = nn.Embedding(movies_count, embedding_size)
        self.um_dense = nn.Linear(2*embedding_size, embedding_size)
        self.um_dense_2 = nn.Linear(embedding_size, 10)

        self.movie_features = nn.Linear(features_count, 10)
        self.user_context = nn.Linear(context_count, context_count)

        self.final_fc1 = nn.Linear(10+10+2, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x_user = x['user'] 
        x_movie = x['movie']
        x_movie_features = x['movie_features']
        x_context_features = x['user_context']

        emb_user = self.user_embedding(x_user)
        emb_movie = self.movie_embedding(x_movie)   
        um_cat = torch.cat((emb_user, emb_movie), 1)
        um_fc1 = F.relu(self.um_dense(um_cat))
        um_fc2 = F.relu(self.um_dense_2(um_fc1))

        feat_fc = F.relu(self.movie_features(x_movie_features.float()))
        con_fc = F.relu(self.user_context(x_context_features.float()))

        um_cat_2 = torch.cat((um_fc2, feat_fc, con_fc), 1)
        final_fc = self.final_fc1(um_cat_2)
        
        output_layer = torch.sigmoid(self.output(final_fc))
        
        return output_layer

    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds.float(), y.float())

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x_val, y_val = batch
        preds_val = self(x_val)
        loss_val = self.loss_fn(preds_val.float(), y_val.float())

        self.log("val_loss", loss_val, on_epoch=True, on_step=True)
        return loss_val
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)