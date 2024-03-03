import logging

from src.evaluation.f1_score_seq2seq import F1ScoreSeq2Seq
from src.evaluation.slots import Slots
from src.utils.config import (NULL_ASPECT_TERM, SEPARATOR_SENTENCES, CATEGORY_SENTIMENT_SEPARATOR,
                              SENTIMENT_TERM_SEPARATOR)


class Evaluator:
    """Evaluator for SemEval 2016 task 5 dataset. Used to evaluate sequence-to-sequence model."""

    def process_batch_for_evaluation(
            self,
            decoded_predictions: list[str],
            labels: list[str],
            acd_f1_score: F1ScoreSeq2Seq,
            ate_f1_score: F1ScoreSeq2Seq,
            acte_f1_score: F1ScoreSeq2Seq,
            tasd_f1_score: F1ScoreSeq2Seq,
    ) -> None:
        """
        Process batch for evaluation. Convert predictions and labels to slots and update metrics.

        :param decoded_predictions: decoded predictions
        :param labels: labels
        :param acd_f1_score: acd f1 score
        :param ate_f1_score: ate f1 score
        :param acte_f1_score: acte f1 score
        :param tasd_f1_score: tasd f1 score
        :return: None
        """
        for decoded_prediction, label in zip(decoded_predictions, labels):
            logging.info("Example:")
            logging.info("Decoded prediction: %s", str(decoded_prediction))
            logging.info("Label: %s", str(label))
            slots_predictions = self._retrieve_slots(decoded_prediction)
            slots_labels = self._retrieve_slots(label)
            acd_f1_score.update(predictions=slots_predictions.acd, labels=slots_labels.acd)
            ate_f1_score.update(predictions=slots_predictions.ate, labels=slots_labels.ate)
            acte_f1_score.update(predictions=slots_predictions.acte, labels=slots_labels.acte)
            tasd_f1_score.update(predictions=slots_predictions.tasd, labels=slots_labels.tasd)

    def _retrieve_slots(self, sample: str) -> Slots:
        """
        Retrieve slots from data sample.

        :param sample: data sample
        :return: slots from sample
        """
        acd = set()
        ate = set()
        acte = set()
        tasd = set()

        # Get slots if sample is not empty
        try:
            self._parse_sample(sample=sample, acd=acd, ate=ate, acte=acte, tasd=tasd)
        except IndexError as e:
            logging.error("ValueError: %s", str(e))
            return Slots(acd=acd, ate=ate, acte=acte, tasd=tasd)

        # Remove "NULL" from ate if it exists
        ate.discard(NULL_ASPECT_TERM)

        slots = Slots(acd=acd, ate=ate, acte=acte, tasd=tasd)

        return slots

    def _parse_sample(self, sample: str, acd: set, ate: set, acte: set, tasd: set) -> None:
        """
        Parse sample to retrieve slots.

        :param sample: data sample
        :param acd: acd contains aspect categories
        :param ate: ate contains aspect terms
        :param acte: acte contains aspect categories and aspect terms
        :param tasd: tasd contains aspect categories, aspect terms and sentiment
        :return: None
        """
        if sample.strip():
            sentences = self._retrieve_sentences(sample)

            for sentence in sentences:
                try:
                    aspect_category, ote, sentiment = self._parse_sentence(sentence)
                    acd.add(aspect_category)
                    ate.add(ote)
                    # Add aspect category ote tuple to ACTE
                    acte.add((aspect_category, ote))
                    # Add aspect category, ote, sentiment tuple to TASD
                    tasd.add((aspect_category, ote, sentiment))

                except ValueError as e:
                    logging.error("ValueError: %s - %s", str(sentence), str(e))

    def _retrieve_sentences(self, sample: str) -> list[str]:
        """
        Retrieve sentences from sample.

        :param sample: data sample
        :return: list of sentences
        """
        sentences = [sent.strip() for sent in sample.split(SEPARATOR_SENTENCES)]
        return sentences

    def _parse_sentence(self, sentence: str) -> tuple[str, str, str]:
        """
        Parse sentence to retrieve aspect category, aspect term and sentiment.

        :param sentence: sentence to parse
        :return: aspect category, aspect term and sentiment
        """
        aspect_category_sentiment, ote = sentence.split(f"{SENTIMENT_TERM_SEPARATOR}")
        aspect_category, sentiment = aspect_category_sentiment.split(f" {CATEGORY_SENTIMENT_SEPARATOR} ")
        aspect_category = aspect_category.strip()
        sentiment = sentiment.strip()
        ote = ote.strip()
        return aspect_category, ote, sentiment
