import logging
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding

from src.utils.config import (CATEGORY_TO_LABEL_MAPPING, E2E_MAPPING, ATE_MAPPING, SENTIMENT2LABEL,
                              TEXT_LABEL_SEPARATOR, CATEGORY_MAPPING_TO_CZECH, LOSS_IGNORE_INDEX)
from src.utils.tasks import Task


class SemEval2016DatasetSmallerTasks(Dataset):
    """Dataset for smaller ABSA tasks."""

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len_text: int,
            task: Task,
    ) -> None:
        """
        Initialize dataset for smaller ABSA tasks with given arguments.

        :param data_path: path to the data file
        :param tokenizer:  tokenizer
        :param max_seq_len_text: maximum length of the text sequence
        :param task: task to solve
        """

        self._data_path = data_path
        self._max_seq_len_text = max_seq_len_text
        self._tokenizer = tokenizer
        self._task = task

        self._encoded_inputs = []
        self._labels = []

        self._convert_label_to_czech = False
        tok_name = tokenizer.name_or_path.lower()
        if "czert" in tok_name or "robeczech" in tok_name or "fernet" in tok_name or "czech" in tok_name:
            self._convert_label_to_czech = True

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
        - labels: label text

        :param index: index of the item
        :return: dictionary containing input text ids, input attention mask, label ids and label for item at the given
            index
        """
        source_ids = self._encoded_inputs[index].input_ids.squeeze()
        target_ids = torch.tensor([])

        source_mask = self._encoded_inputs[index].attention_mask.squeeze()

        return {
            "input_text_ids": source_ids,
            "input_attention_mask": source_mask,
            "labels_ids": target_ids,
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
            opinions = sentence_element.find("Opinions")
            if opinions is None or opinions.find("Opinion") is None or len(list(opinions.iter("Opinion"))) == 0:
                continue

            self._process_opinions(opinions=opinions, text=text)

        # Print example of first sentence as tokens and converted back to text
        first_padding_idx = self._encoded_inputs[0].attention_mask.argmin()
        if first_padding_idx == 0:
            first_padding_idx = self._encoded_inputs[0].attention_mask.shape[1]
        first_example_input_ids = self._encoded_inputs[0].input_ids[0]
        logging.info("Example of first sentence token ids: %s", str(first_example_input_ids[:first_padding_idx]))
        logging.info(
            "Example of first sentence: %s", str(self._tokenizer.decode(first_example_input_ids[:first_padding_idx]))
            )
        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Number of samples: %d", len(self._encoded_inputs))

    def _process_opinions(
            self,
            opinions: ET.Element,
            text: str,
    ) -> None:
        """
        Process opinions.

        :param opinions: opinions
        :param text: text
        :return: None
        """
        labels_annotation = set()
        for i, opinion in enumerate(opinions.iter("Opinion")):
            opinion_target_expression = opinion.attrib.get("target", None)
            sentiment = opinion.attrib.get("polarity", None)
            category = opinion.attrib.get("category", None)
            from_ = opinion.attrib.get("from", None)
            to = opinion.attrib.get("to", None)
            if opinion_target_expression is None or sentiment is None or category is None:
                continue
            if self._task == Task.APD:
                labels_annotation.add((opinion_target_expression, category, sentiment, i))
            elif self._task == Task.ACD:
                label = CATEGORY_TO_LABEL_MAPPING[category]
                labels_annotation.add(label)
            elif self._task == Task.ATE:
                if opinion_target_expression == "NULL":
                    continue
                # from and to are character indices, convert them to word indices if i consider text.split() as words
                from_ = int(from_)
                to = int(to)
                labels_annotation.add((sentiment, from_, to))
            else:
                from_ = int(from_)
                to = int(to)
                # If the from_ to tuple already exists with different sentiment, remove it and add neutral
                from_to_dict = {(item[1], item[2]): item for item in labels_annotation}
                key = (from_, to)
                if key in from_to_dict:
                    existing_tuple = from_to_dict[key]
                    if existing_tuple[0] != sentiment:
                        labels_annotation.remove(existing_tuple)
                        sentiment = "neutral"
                labels_annotation.add((sentiment, from_, to))

        if self._task == Task.APD:
            # Sort labels by index for purposes of tests
            labels_annotation = sorted(labels_annotation, key=lambda x: x[3])
            for opinion_target_expression, category, sentiment, _ in labels_annotation:
                if sentiment not in SENTIMENT2LABEL:
                    continue
                built_text, label = self._build_input_and_label_for_classification(
                    category=category,
                    opinion_target_expression=opinion_target_expression,
                    sentiment=sentiment,
                    text=text,
                )

                encoded_input = self._tokenize_text(built_text)
                self._encoded_inputs.append(encoded_input)

                self._labels.append(label)

        elif self._task == Task.ACD:
            tokenized_text = self._tokenize_text(text)
            self._encoded_inputs.append(tokenized_text)
            # convert labels to torch tensor, where 1 is when corresponding label is present, 0 otherwise
            labels_acd = [0 for _ in range(len(CATEGORY_TO_LABEL_MAPPING))]
            for label in labels_annotation:
                labels_acd[label] = 1
            self._labels.append(torch.tensor(labels_acd, dtype=torch.float))

        else:
            tokens = self._tokenizer.tokenize(
                text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=self._max_seq_len_text,
            )

            tokenized_text = self._tokenize_text(text, return_offsets_mapping=True)

            offsets = tokenized_text["offset_mapping"][0]

            labels = [0 for _ in range(len(tokens))]

            for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
                # token = self._tokenizer.decode([token_id])
                if token == self._tokenizer.cls_token or token == self._tokenizer.sep_token or token == self._tokenizer.pad_token:
                    labels[i] = LOSS_IGNORE_INDEX
                    continue
                if token.startswith("##"):
                    labels[i] = LOSS_IGNORE_INDEX  # Set the label to -100 for subword tokens that are not the first token
                    continue
                # find if the token is part of the aspect term by iterating through labels and finding if it is within from and to
                for label in labels_annotation:
                    sentiment, from_, to = label
                    if start >= from_ and end <= to:
                        if start == from_:
                            if self._task == Task.ATE:
                                labels[i] = ATE_MAPPING["B"]
                            else:
                                labels[i] = E2E_MAPPING["B-" + sentiment]
                        else:
                            if self._task == Task.ATE:
                                labels[i] = ATE_MAPPING["I"]
                            else:
                                labels[i] = E2E_MAPPING["I-" + sentiment]

            self._encoded_inputs.append(tokenized_text)
            self._labels.append(torch.tensor(labels, dtype=torch.long))

    def _build_input_and_label_for_classification(
            self,
            category: str,
            opinion_target_expression: str,
            sentiment: str,
            text: str,
    ) -> tuple[str, int]:
        """
        Build input and label for classification. Convert category, append converted category and opinion target
        expression to text and append sentiment as label. Return input and label.

        :param category: category
        :param opinion_target_expression: opinion target expression
        :param sentiment: sentiment
        :param text: text
        :return: input and label
        """
        converted_category = self._convert_category(category)
        built_text = f"{text} {TEXT_LABEL_SEPARATOR} {converted_category} {opinion_target_expression}"
        label = SENTIMENT2LABEL[sentiment]
        return built_text, label

    def _convert_category(self, category: str) -> str:
        """
        Convert category in format 'ENTITY#ATTRIBUTE' to '<Entity> <attribute>'.

        :param category: category in format 'ENTITY#ATTRIBUTE'
        :return: category in format '<Entity> <attribute>'
        """
        if self._convert_label_to_czech:
            if category not in CATEGORY_MAPPING_TO_CZECH:
                logging.error("category %s not in CATEGORY_MAPPING", category)
                exit(1)
            return CATEGORY_MAPPING_TO_CZECH[category]
        if "#" in category:
            split = category.split("#")
            category = f"{split[0].title()} {split[1].lower()}"
        return category

    def _tokenize_text(self, text: str, return_offsets_mapping: bool = False) -> BatchEncoding:
        """
       Tokenize text and return token ids and attention mask.

       :param text: text to tokenize
       :param return_offsets_mapping: if True, return offsets mapping
       :return: dictionary containing token IDs and attention mask (and possibly offsets mapping)
        """
        return self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_seq_len_text,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=return_offsets_mapping,
        )
