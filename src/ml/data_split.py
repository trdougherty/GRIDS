import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from src.ml.data_collector import DataCollector

class DataSplit(pl.LightningDataModule):
    def __init__(self, data_location:str = "") -> None:
        super().__init__()
        self.data_location = data_location

    def setup(self, stage):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        
        image_data = DataCollector(pkl_file=self.data_location, transform=transform)

        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        train_set, val_set, test_set = random_split(image_data, (train_size, val_size, test_size))
        self.batch_size = 32

        if stage == 'fit' or stage is None:
            self.train_set = train_set
            self.val_set = val_set
        if stage == 'test':
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8)