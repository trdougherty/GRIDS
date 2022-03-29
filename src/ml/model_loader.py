import torch.nn as nn
import torch
import pytorch_lightning as pl

class ModelLoader(pl.LightningModule):
    def __init__(self, lr: float = 2e-3, num_workers: int = 8, batch_size: int = 16):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.leaky = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.5)

          # FC Tabular
        self.ln6 = nn.Linear(2, 5)
        self.ln7 = nn.Linear(5, 10)
       
        # Final
        self.ln10 = nn.Linear(10, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters, lr=(self.lr))

    def forward(self, tab):
        tab = self.ln6(tab)
        tab = self.leaky(tab)
        tab = self.ln7(tab)
        tab = self.leaky(tab)

        return self.ln10(tab)

    def training_step(self, batch, batch_idx):
        tabular, y = batch

        criterion = nn.SmoothL1Loss()

        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        tabular, y = batch

        criterion = nn.SmoothL1Loss()
        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        tabular, y = batch

        loss = nn.SmoothL1Loss()
        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()
        # y_pred = self.forward(z19,z20,tabular)
        test_loss = loss(y_pred, y)

        return {"test_loss": test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}