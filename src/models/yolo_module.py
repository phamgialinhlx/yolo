from typing import Any

import torch
import torch.optim as optim
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import hydra
import omegaconf
import pyrootutils

class YOLOLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        lr,
        weight_decay,
    ):
        super().__init__()
        self.net = net
        self.save_hyperparameters(logger=False)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):
        print(batch)
        print(batch.shape)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # img, label = batch
        loss, preds, targets = self.model_step(batch)
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(
            self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

if __name__ == "__main__":
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "yolo.yaml")
    _ = hydra.utils.instantiate(cfg)
    print(_.net)