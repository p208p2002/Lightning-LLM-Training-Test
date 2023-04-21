from torch.utils.data import random_split,DataLoader
from .FooDataset import FooDataset
import lightning.pytorch as pl

class FooDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        dataset = FooDataset()

        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = int(len(dataset) * 0.1)
        test_set_size = len(dataset) - (train_set_size + valid_set_size)
        train_set, valid_set, test_set = random_split(
            dataset, [train_set_size, valid_set_size, test_set_size])

        self.train = train_set
        self.valid = valid_set
        self.test = test_set

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

