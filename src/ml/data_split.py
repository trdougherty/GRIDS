import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from ml.data_collector import DataCollector

class DataSplit(pl.LightningDataModule):
    """Takes the data location and splits it nicely into train, validation, and test"""
    def __init__(self, num_workers:int = 8, batch_size:int = 32, data_locations:dict = {}, **kwargs) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_locations = data_locations

    def setup(self, stage):
        data_module = DataCollector(**self.data_locations)
        train_size = int(0.80 * len(data_module))
        val_size = int((len(data_module) - train_size))

        train_set, val_set = random_split(data_module, (train_size, val_size))

        if stage == 'fit' or stage is None:
            self.train_set = train_set
            self.val_set = val_set
        else:
            self.test_set = None

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
        # return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)