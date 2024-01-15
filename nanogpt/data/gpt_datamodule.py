import logging
from pathlib import Path

import lightning
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, IterableDataset, IterDataPipe
from torchtext.datasets import DATASETS, WikiText2
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def _data_gen(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    yield from loader


def _train_tokenizer(dataset, vocab_size, whitespace_tok_enabled=True):
    tokenizer = Tokenizer(BPE())
    if whitespace_tok_enabled:
        tokenizer.pre_tokenizer = Whitespace()
    tokenizer_trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train_from_iterator(_data_gen(dataset=dataset), trainer=tokenizer_trainer)

    return tokenizer


def _collate_func(item, device=None):
    x = np.array(item)
    return torch.tensor(x[:, 0, :], dtype=torch.long, device=device), torch.tensor(
        x[:, 1, :], dtype=torch.long, device=device
    )


class NanoGPTDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "WikiText2",
        vocab_size: int = 500,
        block_size: int = 10,
        batch_size: int = 8,
        min_len: int = 100,
        force_tokenizer_retrain: bool = False,
        tokenizer_template: str = "data/tokenizer/tokenizer_vocab_{vocab_size}.json",
        whitespace_tok_enabled: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False, ignore=["force_tokenizer_retrain", "tokenizer_template"]
        )

        self.force_tokenizer_retrain = force_tokenizer_retrain
        self.tokenizer_path = Path(tokenizer_template.format(vocab_size=vocab_size))

        self.train, self.val, self.test = None, None, None

        self.batch_size_per_device = batch_size

    def setup(self, stage: str) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_path))

        def preprocess(dataset: IterableDataset):
            return (
                dataset.map(lambda x: self.tokenizer.encode(x))
                .filter(lambda x: len(x) > self.hparams.min_len)
                .flatmap(
                    lambda text: [
                        text[i : i + self.hparams.block_size + 1]
                        for i in range(
                            0, len(text) - self.hparams.block_size + 2, self.hparams.block_size + 1
                        )
                    ]
                )
                .filter(lambda x: len(x) == self.hparams.block_size + 1)
                .map(lambda x: (x[:-1], x[1:]))
            )

        # if stage == "fit":
        raw_datasets: IterDataPipe = WikiText2()
        self.train, self.val, self.test = map(preprocess, raw_datasets)

    def prepare_data(self) -> None:
        datasets = DATASETS[self.hparams.dataset_name]()
        # device
        for dataset in datasets:
            next(iter(dataset))

        if not self.tokenizer_path.exists() or self.force_tokenizer_retrain:
            logger.info("Training tokenizer")
            raw_train, _, _ = datasets
            tokenizer_slow = _train_tokenizer(
                raw_train,
                vocab_size=self.hparams.vocab_size,
                whitespace_tok_enabled=self.hparams.whitespace_tok_enabled,
            )
            logger.info("Saving tokenizer uder %s", self.tokenizer_path)
            self.tokenizer_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.tokenizer_path, "w") as fp:
                fp.write(tokenizer_slow.to_str(pretty=True))

    def train_dataloader(self):
        # TODO check if collate_fn is needed
        # .manual_seed(self.config.random_seed)
        return DataLoader(
            self.train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_func,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_func,
        )


if __name__ == "__main__":
    _ = NanoGPTDataModule()
