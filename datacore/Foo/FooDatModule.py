from torch.utils.data import DataLoader
from .FooDataset import FooDataset
import lightning.pytorch as pl
from config import get_args

args = get_args()

class FooDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = args.batch_size
        self.train = FooDataset()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

