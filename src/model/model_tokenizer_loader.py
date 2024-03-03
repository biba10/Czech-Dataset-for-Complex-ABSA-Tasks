import pytorch_lightning as pl
from transformers import (PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification)

from src.evaluation.evaluation import Evaluator
from src.model.model_classification import ABSAModelClassification
from src.model.model_generative import ABSAModelGenerative
from src.model.model_multilabel import ABSAModelMultilabelClassification
from src.model.model_token_classification import ABSAModelTokenClassification
from src.utils.config import SENTIMENT2LABEL, CATEGORY_TO_LABEL_MAPPING, ATE_MAPPING, E2E_MAPPING
from src.utils.tasks import Task


def load_model_and_tokenizer(
        model_path: str,
        model_max_length: int,
        max_seq_length_label: int,
        optimizer: str,
        learning_rate: float,
        beam_size: int,
        task: Task,
) -> tuple[pl.LightningModule, PreTrainedTokenizer]:
    """
    Load model and tokenizer from path. Add special tokens to tokenizer.

    :param model_path: path to pre-trained model or shortcut name
    :param model_max_length: maximal length of the sequence
    :param max_seq_length_label: maximal length of the label
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param beam_size: beam size
    :param task: task
    :return: model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length, use_fast=True)

    if "t5" in model_path or "bart" in model_path:
        model = _load_generative_model(
            model_path=model_path,
            tokenizer=tokenizer,
            optimizer=optimizer,
            learning_rate=learning_rate,
            beam_size=beam_size,
            max_seq_length=max_seq_length_label,
        )
    elif task is not None and (task == Task.ATE or task == task.E2E):
        model = _load_model_for_token_classification(
            model_path=model_path,
            learning_rate=learning_rate,
            task=task,
        )
    elif task is not None and task == Task.ACD:
        model = _load_model_for_multilabel_classification(
            model_path=model_path,
            learning_rate=learning_rate,
        )
    else:
        # APD task and csfd dataset
        model = _load_model_for_classification(
            model_path=model_path,
            learning_rate=learning_rate,
        )

    return model, tokenizer


def _load_model_for_classification(
        model_path: str,
        learning_rate: float,
) -> pl.LightningModule:
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(SENTIMENT2LABEL))

    absa_model = ABSAModelClassification(
        model=model,
        learning_rate=learning_rate,
    )

    return absa_model


def _load_model_for_multilabel_classification(
        model_path: str,
        learning_rate: float,
) -> pl.LightningModule:
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(CATEGORY_TO_LABEL_MAPPING))

    absa_model = ABSAModelMultilabelClassification(
        model=model,
        learning_rate=learning_rate,
    )

    return absa_model


def _load_model_for_token_classification(
        model_path: str,
        learning_rate: float,
        task: Task,
) -> pl.LightningModule:
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(ATE_MAPPING) if task == Task.ATE else len(E2E_MAPPING),
    )

    absa_model = ABSAModelTokenClassification(
        model=model,
        learning_rate=learning_rate,
        task=task,
    )

    return absa_model


def _load_generative_model(
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        optimizer: str,
        learning_rate: float,
        beam_size: int,
        max_seq_length: int,
) -> pl.LightningModule:
    """
    Load generative model from path. Add special tokens to tokenizer.

    :param model_path: path to pre-trained model or shortcut name
    :param tokenizer: pre-trained tokenizer
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param beam_size: beam size
    :param max_seq_length: maximum sequence length for generation
    :return: model
    """
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    evaluator = Evaluator()

    absa_model = ABSAModelGenerative(
        learning_rate=learning_rate,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        beam_size=beam_size,
        evaluator=evaluator,
        max_seq_length=max_seq_length,
    )

    return absa_model
