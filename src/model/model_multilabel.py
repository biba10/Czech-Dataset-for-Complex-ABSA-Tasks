import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics.classification import MultilabelF1Score
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.utils.config import CATEGORY_TO_LABEL_MAPPING


class ABSAModelMultilabelClassification(pl.LightningModule):
    """Classification model for Aspect Based Sentiment Analysis."""

    def __init__(
            self,
            learning_rate: float,
            model: PreTrainedModel,
    ) -> None:
        """
        Initialize the model.

        :param learning_rate: learning rate
        :param model: pre-trained model (expects a classification head)
        """
        super().__init__()
        self._learning_rate = learning_rate
        self._model = model
        self._acd_f1_score = MultilabelF1Score(num_labels=len(CATEGORY_TO_LABEL_MAPPING), average="micro")

        self.save_hyperparameters(ignore=["model"])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
    ) -> SequenceClassifierOutput:
        """
        Perform forward pass through the model.

        :param input_ids: input ids
        :param attention_mask: attention mask
        :param labels: labels
        :return: model output
        """
        output = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform training step for a single batch. Compute the loss and log it.

        :param batch: batch
        :param batch_idx: batch index
        :return: loss
        """
        out = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step for a single batch. Compute the loss and update the metric. Log the loss and metric.

        :param batch: batch
        :param batch_idx: batch index
        :return: validation loss
        """
        loss = self._compute_loss_and_update_metric(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("acd_f1", self._acd_f1_score, prog_bar=True, sync_dist=True)

        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform test step for a single batch. Compute the loss and update the metric. Log the loss and metric.

        :param batch: batch
        :param batch_idx: batch index
        :return: test loss
        """
        loss = self._compute_loss_and_update_metric(batch)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acd_f1", self._acd_f1_score, prog_bar=True, sync_dist=True)

        return {"test_loss": loss}

    def _compute_loss_and_update_metric(self, batch):
        output = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels"],
        )
        loss = output.loss
        self._acd_f1_score.update(output.logits, batch["labels"])
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer.

        :return: optimizer
        """
        # AdamW
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
        )
        return optimizer
