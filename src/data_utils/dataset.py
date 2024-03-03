import logging
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding

from src.utils.config import (SENTIMENT2OPINION, LOSS_IGNORE_INDEX, SENTIMENT2LABEL, SEPARATOR_SENTENCES,
                              CATEGORY_SENTIMENT_SEPARATOR, SENTIMENT_TERM_SEPARATOR)


class SemEval2016Dataset(Dataset):
    """Dataset for ABSA dataset for Seq2Seq models and APD task."""

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len_text: int,
            max_seq_len_label: int,
    ) -> None:
        """
        Initialize dataset for SemEval 2016 task 5 dataset with given arguments.

        :param data_path: path to the data file
        :param tokenizer:  tokenizer
        :param max_seq_len_text: maximum length of the text sequence
        :param max_seq_len_label: maximum length of the label sequence
        """

        self._data_path = data_path
        self._max_seq_len_text = max_seq_len_text
        self._max_seq_len_label = max_seq_len_label
        self._tokenizer = tokenizer

        self._encoded_inputs = []
        self._encoded_labels = []
        self._labels = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._encoded_inputs)

    def __getitem__(self, index: int) -> dict:
        """
        Return the dictionary for item at the given index.
        Dictionary contains the following keys:
        - input_text_ids: token ids of the text sequence
        - input_attention_mask: attention mask of the text sequence
        - labels_ids: token ids of the label sequence
        - labels_attention_mask: attention mask of the label sequence
        - labels: label text

        :param index: index of the item
        :return: dictionary containing input text ids, input attention mask, label ids, label attention mask and label
        for item at the given index
        """
        source_ids = self._encoded_inputs[index].input_ids.squeeze()
        target_ids = self._encoded_labels[index].input_ids.squeeze()

        source_mask = self._encoded_inputs[index].attention_mask.squeeze()
        target_mask = self._encoded_labels[index].attention_mask.squeeze()

        return {
            "input_text_ids": source_ids,
            "input_attention_mask": source_mask,
            "labels_ids": target_ids,
            "labels_attention_mask": target_mask,
            "labels": self._labels[index],
        }

    def _load_data(self) -> None:
        """
        Load data from the dataset file. Convert text and label into token IDs and attention masks.

        :return: None
        """
        tree = ET.parse(self._data_path)

        root = tree.getroot()

        sentence_elements = root.iter("sentence")

        for sentence_element in sentence_elements:
            text_element = sentence_element.find("text")
            if text_element is None:
                continue
            text = text_element.text
            if not text:
                continue
            labels = []

            opinions = sentence_element.find("Opinions")
            if opinions is None or opinions.find("Opinion") is None or len(list(opinions.iter("Opinion"))) == 0:
                continue

            self._process_opinions(opinions=opinions, labels=labels)

            labels_joined = f" {SEPARATOR_SENTENCES} ".join(labels)
            # Add empty label token if labels_joined are empty
            if not labels_joined:
                labels_joined = ""

            self._add_sample(text=text, label=labels_joined, label_to_encode=labels_joined)

        # Print example of first sentence as tokens and converted back to text
        first_padding_idx = self._encoded_inputs[0].attention_mask.argmin()
        if first_padding_idx == 0:
            first_padding_idx = self._encoded_inputs[0].attention_mask.shape[1]
        first_example_input_ids = self._encoded_inputs[0].input_ids[0]
        logging.info("Example of first sentence token ids: %s", str(first_example_input_ids[:first_padding_idx]))
        logging.info(
            "Example of first sentence: %s",
            str(self._tokenizer.decode(first_example_input_ids[:first_padding_idx]))
        )

        first_padding_idx = self._encoded_labels[0].attention_mask.argmin()
        if first_padding_idx == 0:
            first_padding_idx = self._encoded_labels[0].attention_mask.shape[1]
        logging.info(
            "Example of first label token ids: %s", str(self._encoded_labels[0].input_ids[0][:first_padding_idx])
        )
        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Number of samples: %d", len(self._encoded_inputs))

    def _process_opinions(
            self,
            opinions: ET.Element,
            labels: list[str],
    ) -> None:
        """
        Process opinions.

        :param opinions: opinions
        :param labels: list of labels, used for seq2seq models
        :return: None
        """
        for opinion in opinions.iter("Opinion"):
            opinion_target_expression = opinion.attrib.get("target", None)
            sentiment = opinion.attrib.get("polarity", None)
            category = opinion.attrib.get("category", None)
            if opinion_target_expression is None or sentiment is None or category is None:
                continue
            if sentiment not in SENTIMENT2LABEL:
                continue
            converted_label = self._build_label(
                opinion_target_expression=opinion_target_expression,
                sentiment=sentiment,
                category=category,
            )
            labels.append(converted_label)

    def _add_sample(self, text: str, label: str | int, label_to_encode: str) -> None:
        """
        Encode input text add it to the list.
        Append label to the list of labels.
        Encode label to encode it and append it to the list of labels to encode.
        Ensure that the label is not computed on pad tokens.

        :param text: text to encode
        :param label: label
        :param label_to_encode: label to encode
        :return: None
        """
        encoded_input = self._tokenize_text(text, self._max_seq_len_text)
        self._encoded_inputs.append(encoded_input)

        self._labels.append(label)

        encoded_label = self._tokenize_text(label_to_encode, self._max_seq_len_label)

        # Ignore computing loss on pad tokens
        encoded_label.input_ids[encoded_label.input_ids == self._tokenizer.pad_token_id] = LOSS_IGNORE_INDEX
        self._encoded_labels.append(encoded_label)

    def _tokenize_text(self, text: str, max_length: int) -> BatchEncoding:
        """
        Tokenize text and return token ids and attention mask.

        :param text: text to tokenize
        :param max_length: maximum length of the tokenized text
        :return: dictionary containing token IDs and attention mask
        """
        encoded_inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return encoded_inputs

    def _build_label(
            self,
            opinion_target_expression: str,
            sentiment: str,
            category: str,
    ) -> str:
        """
        Build a label text for the given opinion target expression, sentiment and category.

        The label text is built as follows:
        - category is converted into '<Entity> <attribute>'
        - sentiment is added to converted category as '<Entity> <attribute> is <sentiment>'
        - final label is built as '<Entity> <attribute> is <sentiment>, given the expression <OTE>'

        :param opinion_target_expression: opinion target expression (OTE)
        :param sentiment: sentiment ('positive', 'negative', 'neutral')
        :param category: category in format 'ENTITY#ATTRIBUTE'
        :return: label text
        """
        category = _convert_category(category)

        sentiment = SENTIMENT2OPINION[sentiment]

        label = f"{category} {CATEGORY_SENTIMENT_SEPARATOR} {sentiment}{SENTIMENT_TERM_SEPARATOR} {opinion_target_expression}"

        return label


def _convert_category(category: str) -> str:
    """
    Convert category in format 'ENTITY#ATTRIBUTE' to '<Entity> <attribute>'.

    :param category: category in format 'ENTITY#ATTRIBUTE'
    :return: category in format '<Entity> <attribute>'
    """
    if "#" in category:
        split = category.split("#")
        category = f"{split[0].title()} {split[1].lower()}"
    return category
