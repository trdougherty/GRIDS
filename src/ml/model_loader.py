import torch.nn as nn
import torch
import pytorch_lightning as pl
import wandb

from ml.errors import cvstd, cvrmse, nmbe 
# from ml.models import graph_mean

class ModelLoader(pl.LightningModule):
    def __init__(self, input_size:int, output_size:int = 2, lr: float = 2e-3, batch_size: int = 16, **kwargs):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.leaky = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.5)
        self.loss_function = nn.SmoothL1Loss()


        self.neuralnet = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 200),
            nn.BatchNorm1d(200),
            nn.Linear(200,200),
            nn.BatchNorm1d(200),
            nn.Linear(200,200),
            nn.BatchNorm1d(200),
            nn.Linear(200,2)
        )

    def _loss(self, y_pred, y):
        y_pred = y_pred.float()
        y = y.float()
        return self.loss_function(y_pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def forward(self, x):
        return self.neuralnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self._loss(y_pred, y)

        self.log({"train/cvstd": cvstd(y_pred, y)})
        self.log({"train/cvrmse": cvrmse(y_pred, y)})
        self.log({"train/nmbe": nmbe(y_pred, y)})

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self._loss(y_pred, y)

        self.log({"val/cvstd": cvstd(y_pred, y)})
        self.log({"val/cvrmse": cvrmse(y_pred, y)})
        self.log({"val/nmbe": nmbe(y_pred, y)})

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # y_pred = self.forward(z19,z20,tabular)
        loss = self._loss(y_pred, y)

        self.log({"test/cvstd": cvstd(y_pred, y)})
        self.log({"test/cvrmse": cvrmse(y_pred, y)})
        self.log({"test/nmbe": nmbe(y_pred, y)})

        return loss