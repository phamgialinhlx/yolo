import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, Optional, Tuple
from lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from PIL import Image
from models.utils.yolo_utils import yolo_box

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

class VOCDataset(VOCDetection):
    def __getitem__(self, index: int) -> Dict[str, Any]:    
        img, target = super().__getitem__(index)
        return img, yolo_box(target, CLASSES)
class VOCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        train_val_split: Tuple[int, int] = (9000, 2540),
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448), antialias=True),
            ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    @property
    def num_classes(self):
        return 20

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        VOCDetection(self.hparams.data_dir, image_set="trainval", download=True)
        VOCDetection(self.hparams.data_dir, year="2007", image_set="test", download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_test = VOCDataset(self.hparams.data_dir, year="2007", image_set="test", transform=self.transforms)
            self.data_train, self.data_val = random_split(
                dataset=VOCDataset(self.hparams.data_dir, image_set="trainval", transform=self.transforms),
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda x: x,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,

        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda x: x,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda x: x,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
if __name__ == "__main__":
    _ = VOCDataModule()
    # _.prepare_data()
    _.setup()
    train_loader = _.train_dataloader()
    val_loader = _.val_dataloader()
    test_loader = _.test_dataloader()
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    # from IPython import embed
    # embed()
    for x in next(iter(train_loader)):
        print(x[0].shape)
        break
    # image, label = 
    # print(image.shape)
    # print(label)