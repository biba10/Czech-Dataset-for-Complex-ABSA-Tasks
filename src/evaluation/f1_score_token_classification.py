import torch
from torchmetrics import Metric


class F1ScoreTokenClassification(Metric):
    """F1 score for evaluation of token classification. F1 score is calculated as 2 * (precision * recall) / (precision + recall)."""
    full_state_update = False

    def __init__(self) -> None:
        """Initialize the F1 score metric."""
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
            self,
            predictions: list[list[tuple[int, int]]] | list[list[tuple[int, int, str]]],
            labels: list[list[tuple[int, int]]] | list[list[tuple[int, int, str]]],
    ) -> None:
        """
        Update the metric with given predictions and labels.

        :param predictions: predictions
        :param labels: labels
        :return: None
        """
        for prediction, label in zip(predictions, labels):
            for pred in prediction:
                if pred in label:
                    self.tp += 1
                else:
                    self.fp += 1

            for lab in label:
                if lab not in prediction:
                    self.fn += 1

    def calculate_precision(self) -> torch.Tensor:
        """
        Calculate precision. Precision is calculated as tp / (tp + fp).

        :return: precision
        """
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)
        precision = self.tp / (self.tp + self.fp)
        return precision

    def calculate_recall(self) -> torch.Tensor:
        """
        Calculate recall. Recall is calculated as tp / (tp + fn).

        :return: recall
        """
        if self.tp + self.fn == 0:
            return torch.tensor(0.0)
        recall = self.tp / (self.tp + self.fn)
        return recall

    def compute(self) -> torch.Tensor:
        """
        Calculate F1 score. F1 score is calculated as 2 * (precision * recall) / (precision + recall).

        :return: F1 score
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        if precision + recall == 0:
            return torch.tensor(0.0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
