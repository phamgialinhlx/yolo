import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any

import torch
import torch.optim as optim
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import hydra
import omegaconf
from models.utils.yolo_utils import yolo_box, get_bboxes, mean_average_precision


class YOLOLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        lr,
        weight_decay,

    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net
        self.loss = loss

        self.train_loss = MeanMetric()
        self.mAP = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):
        print(len(batch))
        img = torch.stack([x[0] for x in batch])
        label = torch.stack([x[1] for x in batch])
        out = self.forward(img)
        loss = self.loss(out, label)
        preds = torch.argmax(out, dim=1)
        return loss, preds, label

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # img, label = batch
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        pred_boxes, target_boxes = get_bboxes(
            batch, self.net, iou_threshold=0.1, threshold=0.1, device=batch[0].device
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.1, box_format="midpoint"
        )

        self.mAP(mean_avg_prec)

        self.log("val/mAP", self.mAP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
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