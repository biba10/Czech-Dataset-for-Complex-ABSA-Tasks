import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import F1Score
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.evaluation.f1_score_token_classification import F1ScoreTokenClassification
from src.utils.config import ATE_MAPPING, E2E_MAPPING, REVERSE_ATE_MAPPING, REVERSE_E2E_MAPPING
from src.utils.tasks import Task


class ABSAModelTokenClassification(pl.LightningModule):
    """Classification model for Aspect Based Sentiment Analysis."""

    def __init__(
            self,
            learning_rate: float,
            model: PreTrainedModel,
            task: Task,
    ) -> None:
        """
        Initialize the model.

        :param learning_rate: learning rate
        :param model: pre-trained model (expects a classification head)
        :param task: task
        """
        super().__init__()
        self._learning_rate = learning_rate
        self._model = model
        self._task = task
        # self._f1_score = F1Score(
        #     task="multiclass", average="micro",
        #     num_classes=len(ATE_MAPPING) if self._task == Task.ATE else len(E2E_MAPPING)
        # )
        self._f1_score = F1ScoreTokenClassification()

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
        self.log(f"{self._task}_f1", self._f1_score, prog_bar=True, sync_dist=True)

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
        self.log(f"test_{self._task}_f1", self._f1_score, prog_bar=True, sync_dist=True)

        return {"test_loss": loss}

    def _compute_loss_and_update_metric(self, batch):
        output = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels"],
        )
        loss = output.loss
        argmax_predictions = torch.argmax(output.logits, dim=-1).tolist()
        # compute only for those, where batch["labels"] != -100
        predictions = []
        labels = []
        for argmax_prediction, batch_labels in zip(argmax_predictions, batch["labels"].tolist()):
            preds = []
            labs = []
            for prediction, label in zip(argmax_prediction, batch_labels):
                if label != -100:
                    preds.append(prediction)
                    labs.append(label)
            predictions.append(preds)
            labels.append(labs)
        # predictions = predictions[batch["labels"] != -100].tolist()
        # labels = batch["labels"][batch["labels"] != -100].tolist()
        # make it a list
        if self._task == Task.ATE:
            converted_predictions = convert_bio_to_ate(predictions)
            converted_labels = convert_bio_to_ate(labels)
        else:
            converted_predictions = convert_bio_to_e2e(predictions)
            converted_labels = convert_bio_to_e2e(labels)
        self._f1_score.update(converted_predictions, converted_labels)
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

def convert_bio_to_ate(bio_numbers: list[list[[int]]]) -> list[list[[tuple[int, int]]]]:
    """
    Convert BIO tagging to ATE format.

    :param bio_numbers: list of lists of BIO tags in number format
    :return: list of lists of ATE with beginning and end indices
    """
    # Use REVERSE_ATE_MAPPING to map from numbers to word labels
    bio = [[REVERSE_ATE_MAPPING[number] for number in sequence] for sequence in bio_numbers]
    absa_sequences = []
    for sequence in bio:
        beginning = -1
        end = -1
        absa_sequence = []
        for i, tag in enumerate(sequence):
            if tag == "B":
                if beginning != -1 and end != -1:
                    absa_sequence.append((beginning, end))
                beginning = i
                end = i
            elif tag == "I":
                end = i
            elif tag == "O":
                if beginning != -1 and end != -1:
                    absa_sequence.append((beginning, end))
                    beginning = -1
                    end = -1
        if beginning != -1 and end != -1:
            absa_sequence.append((beginning, end))
        absa_sequences.append(absa_sequence)
    return absa_sequences


def convert_bio_to_e2e(bio_numbers: list[list[[int]]]) -> list[list[[tuple[int, int, str]]]]:
    """
    Convert BIO tagging to E2E format.

    :param bio_numbers: list of lists of BIO tags in number format
    :return: list of lists of E2E with beginning and end indices
    """
    bio = [[REVERSE_E2E_MAPPING[number] for number in sequence] for sequence in bio_numbers]
    absa_sequences = []
    for sequence in bio:
        beginning = -1
        end = -1
        sentiments = []
        absa_sequence = []
        for i, tag in enumerate(sequence):
            pos, sentiment = tag.split("-") if "-" in tag else (tag, "O")
            if pos == "B":
                if beginning != -1 and end != -1 and len(set(sentiments)) == 1:
                    absa_sequence.append((beginning, end, sentiments[-1]))
                beginning = i
                end = i
                sentiments = [sentiment]
            elif pos == "I":
                end = i
                sentiments.append(sentiment)
            elif pos == "O":
                if beginning != -1 and end != -1 and len(set(sentiments)) == 1:
                    absa_sequence.append((beginning, end, sentiments[-1]))
                    beginning = -1
                    end = -1
                    sentiments = []
        if beginning != -1 and end != -1 and len(set(sentiments)) == 1:
            absa_sequence.append((beginning, end, sentiments[-1]))
        absa_sequences.append(absa_sequence)
    return absa_sequences




