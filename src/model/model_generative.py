import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from transformers import Adafactor, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.evaluation.f1_score_seq2seq import F1ScoreSeq2Seq
from src.evaluation.evaluation import Evaluator


class ABSAModelGenerative(pl.LightningModule):
    """Generative model for Aspect Based Sentiment Analysis."""

    def __init__(
            self,
            learning_rate: float,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            optimizer: str,
            beam_size: int,
            evaluator: Evaluator,
            max_seq_length: int,
    ) -> None:
        """
        Initialize the model.

        :param learning_rate: learning rate
        :param model: pre-trained model (expecting a Seq2Seq model)
        :param tokenizer: pre-trained tokenizer
        :param optimizer: optimizer
        :param beam_size: beam size
        :param evaluator: evaluator
        :param max_seq_length: maximum sequence length for generation
        """
        super().__init__()

        self._learning_rate = learning_rate
        self._model = model
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self._beam_size = beam_size
        self._max_seq_length = max_seq_length

        self._evaluator = evaluator
        self._avg_f1 = 0.0
        self._acd_f1_score = F1ScoreSeq2Seq()
        self._ate_f1_score = F1ScoreSeq2Seq()
        self._acte_f1_score = F1ScoreSeq2Seq()
        self._tasd_f1_score = F1ScoreSeq2Seq()

        self.save_hyperparameters(ignore=["model"])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_attention_mask: torch.Tensor,
            labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        """
        Perform forward pass through the model.

        :param input_ids: input ids
        :param attention_mask: attention mask
        :param decoder_attention_mask: decoder attention mask
        :param labels: labels
        :return: model output
        """
        output = self._model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute loss for a batch.

        :param batch: batch
        :return: loss
        """
        out = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels_ids"],
            decoder_attention_mask=batch["labels_attention_mask"],
        )
        loss = out.loss
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform training step for a single batch. Compute loss and log it.

        :param batch: batch
        :param batch_idx: batch index
        :return: loss
        """
        loss = self._compute_loss(batch)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _generate_output_and_update_metrics(self, batch: dict) -> torch.Tensor:
        """
        Generate output and predictions, calculate loss, and update metrics for a single batch.

        :param batch: batch
        :return: loss
        """
        loss = self._compute_loss(batch)

        generated_ids = self._model.generate(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            max_length=self._max_seq_length,
            num_beams=self._beam_size,
        )

        decoded_predictions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        gold_labels = batch["labels"]

        self._evaluator.process_batch_for_evaluation(
            decoded_predictions=decoded_predictions,
            labels=gold_labels,
            acd_f1_score=self._acd_f1_score,
            ate_f1_score=self._ate_f1_score,
            acte_f1_score=self._acte_f1_score,
            tasd_f1_score=self._tasd_f1_score,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: validation loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("acd_f1", self._acd_f1_score, prog_bar=True, sync_dist=True)
        self.log("ate_f1", self._ate_f1_score, prog_bar=True, sync_dist=True)
        self.log("acte_f1", self._acte_f1_score, prog_bar=True, sync_dist=True)
        self.log("tasd_f1", self._tasd_f1_score, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform test step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: test loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acd_f1", self._acd_f1_score, prog_bar=True)
        self.log("test_ate_f1", self._ate_f1_score, prog_bar=True)
        self.log("test_acte_f1", self._acte_f1_score, prog_bar=True)
        self.log("test_tasd_f1", self._tasd_f1_score, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer.

        :return: optimizer
        """
        model = self._model
        if self._optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate)
        elif self._optimizer == "adafactor":
            optimizer = Adafactor(
                model.parameters(), lr=self._learning_rate, scale_parameter=False,
                relative_step=False
            )
        else:
            raise ValueError(f"Optimizer {self._optimizer} not implemented.")
        return optimizer
