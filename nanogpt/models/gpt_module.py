from typing import Any

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import MeanMetric

from .components.gpt import NanoGPT


class NanoGPTLitModule(lightning.LightningModule):
    def __init__(
        self,
        net: NanoGPT,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.net = net
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def model_step(self, batch):
        x, targets = batch
        B, T = x.shape
        y_hat = self.net(x)
        loss = torch.nn.functional.cross_entropy(y_hat.view(B * T, -1), targets.reshape(B * T))
        return loss, y_hat, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    # todo on valid end add generation examples

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def forward(self, batch, max_compl=50) -> Any:
        compl = torch.concat(
            [
                batch,
                torch.zeros((*batch.shape[:-1], max_compl - batch.shape[-1]), dtype=torch.long),
            ],
            dim=1,
        )
        last_id = batch.shape[-1]
        for i in range(last_id, max_compl):
            start_id = max(0, i - self.net.block_size + 1)
            end_id = start_id + self.net.block_size
            context = compl[..., start_id:end_id]
            logits = self.net(context)
            proba = torch.nn.functional.softmax(logits[..., i - start_id, :], dim=-1)
            compl[..., i] = torch.multinomial(proba, 1).view(-1)

        return compl

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
