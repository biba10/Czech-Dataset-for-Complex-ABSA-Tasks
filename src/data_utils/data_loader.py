import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.data_utils.dataset import SemEval2016Dataset
from src.data_utils.dataset_smaller_tasks import SemEval2016DatasetSmallerTasks
from src.utils.config import (ABSA_TRAIN, ABSA_TEST, MODE_DEV, DATA_DIR_PATH, ABSA_DEV)
from src.utils.tasks import Task


class ABSADataLoader(pl.LightningDataModule):
    """Data loader for ABSA."""

    def __init__(
            self,
            dataset_path: str,
            batch_size: int,
            tokenizer: PreTrainedTokenizer,
            max_seq_len_text: int,
            max_seq_len_label: int,
            mode: str,
            task: Task | None,
    ) -> None:
        """
        Initialize data loader for ABSA dataset with given arguments.

        :param dataset_path: dataset path
        :param batch_size: train and validation batch size
        :param tokenizer: tokenizer
        :param max_seq_len_text: maximum length of the text sequence
        :param max_seq_len_label: maximum length of the label sequence
        :param mode: mode - 'dev' splits training data to train and dev sets, 'test' uses all training data for training
        :param task: task to solve
        """
        super().__init__()
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_seq_len_text = max_seq_len_text
        self._max_seq_len_label = max_seq_len_label
        self._dataset_path = dataset_path
        self._mode = mode
        self._task = task

    def setup(self, stage=None) -> None:
        """
        Setup data loader.

        :param stage: stage ('fit' for training or 'test' for testing)
        :return: None
        """
        if stage == "fit" or stage is None:
            # Load train dataset
            data_path_train = os.path.join(DATA_DIR_PATH, self._dataset_path, ABSA_TRAIN)
            data_path_dev = os.path.join(DATA_DIR_PATH, self._dataset_path, ABSA_DEV)
            if self._task is None:
                train_dataset = SemEval2016Dataset(
                    data_path=data_path_train,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    max_seq_len_label=self._max_seq_len_label,
                )
                dev_dataset = SemEval2016Dataset(
                    data_path=data_path_dev,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    max_seq_len_label=self._max_seq_len_label,
                )
            else:
                train_dataset = SemEval2016DatasetSmallerTasks(
                    data_path=data_path_train,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    task=self._task,
                )
                dev_dataset = SemEval2016DatasetSmallerTasks(
                    data_path=data_path_dev,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    task=self._task,
                )

            if self._mode == MODE_DEV:
                self._train_dataset = train_dataset
                self._dev_dataset = dev_dataset

                logging.info("Train data length: %d", len(self._train_dataset))
                logging.info("Dev data length: %d", len(self._dev_dataset))
            else:
                # Merge train and dev dataset and use all examples for training
                self._train_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])

            logging.info("Train data all length: %d", len(self._train_dataset))

        # Load test dataset
        if stage == "test" or stage is None:
            data_path_test = os.path.join(DATA_DIR_PATH, self._dataset_path, ABSA_TEST)
            if self._task is None:
                self._test_dataset = SemEval2016Dataset(
                    data_path=data_path_test,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    max_seq_len_label=self._max_seq_len_label,
                )
            else:
                self._test_dataset = SemEval2016DatasetSmallerTasks(
                    data_path=data_path_test,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    task=self._task,
                )
            logging.info("Test data length: %d", len(self._test_dataset))

    def train_dataloader(self) -> DataLoader:
        """
        Get train data loader.

        :return: train data loader
        """
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get dev data loader.

        :return: dev data loader
        """
        return DataLoader(
            self._dev_dataset,
            batch_size=self._batch_size,
            num_workers=0,
        )

    def test_dataloader(self):
        """
        Get test data loader.

        :return: test data loader
        """
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=0,
        )
