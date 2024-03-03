import argparse
import logging

from src.utils.config import (DATASET_OPTIONS, MODE_OPTIONS, ADAFACTOR_OPTIMIZER, ADAMW_OPTIMIZER,
                              MODE_DEV, LANG_CS_REST_M)
from src.utils.tasks import Task


def init_args() -> argparse.Namespace:
    """
    Initialize arguments for the script.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="t5-base",
        help="Path to pre-trained model or shortcut name."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size."
    )
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximal sequence length.")
    parser.add_argument("--max_seq_length_label", type=int, default=256, help="Maximal sequence length for the label.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--dataset_path", type=str, default=LANG_CS_REST_M, help="Dataset path.",
        choices=DATASET_OPTIONS
    )
    parser.add_argument(
        "--optimizer", type=str, choices=[ADAMW_OPTIMIZER, ADAFACTOR_OPTIMIZER], default=ADAMW_OPTIMIZER,
        help="Optimizer."
    )
    parser.add_argument(
        "--mode", type=str, choices=MODE_OPTIONS, default=MODE_DEV,
        help="Mode - 'dev' splits the training data into training and validation sets. The validation set is used for "
             "selecting the best model, which is then evaluated on the test set of the target language. In the case of "
             "different source and target languages, the validation set is taken from the training data of the target "
             "language and the whole training data is used for training."
             "'test' uses whole training dataset from the source language for training and test dataset from the target "
             "language for evaluation."

    )
    parser.add_argument(
        "--checkpoint_monitor", type=str,
        choices=["val_loss", "acte_f1", "acd_f1", "ate_f1", "pd_f1", "tasd_f1", "e2e_f1"], default="val_loss",
        help="Metric based on which the best model will be stored according to the performance on validation data in "
             "'dev' mode"
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches. "
                                                               "It is used when there is insufficient memory for training"
                                                               " for the required effective batch size."
    )
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for beam search decoding.")
    parser.add_argument("--task", type=Task, choices=list(Task), default=None, help="Task.")
    args = parser.parse_args()

    logging.info("Dataset path: %s", args.dataset_path)

    return args
