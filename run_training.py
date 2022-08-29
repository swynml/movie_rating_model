import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.RecommendationModel import RecommendationModel
from dataloader.RecommendationDataset import RecommendationDataset

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recommender_model")

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

features = ['product', 'year', 'Documentary', 'Western', 'Mystery', 'IMAX', 'Drama', 'Biography', 'Film-Noir', 
            'Music', 'Short', 'Thriller', 'Sport', 'Fantasy', 'Family', 'Children', 'Crime', 'Horror', 'Adult', 
            'Animation', 'Comedy', 'History', 'Adventure', 'Romance', 'War', 'Musical', 'Sci-Fi', 'Action']
contexts = ['daytime', 'weekend']

batch_size = 102400

train_data = RecommendationDataset(dataset_type='train',
                                   read_type='pkl',
                                   data_path='data/full_dataset.csv', 
                                   features=features, 
                                   contexts=contexts)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                               num_workers=4, pin_memory=True)

test_data = RecommendationDataset(dataset_type='test',
                                  read_type='pkl',
                                  data_path='data/full_dataset.csv', 
                                  features=features, 
                                  contexts=contexts)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                              num_workers=4, pin_memory=True)

model = RecommendationModel(lr=0.01, 
                            users_count=max(train_data.all_users)+1, 
                            movies_count=max(train_data.all_movies)+1, 
                            embedding_size=50, 
                            features_count=len(features), 
                            context_count=len(contexts))

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=5,
    dirpath='saved_models/',
    filename='recommender_model-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}'
)

trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1, callbacks=[checkpoint_callback])

mlflow.pytorch.autolog()
with mlflow.start_run() as run:
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
