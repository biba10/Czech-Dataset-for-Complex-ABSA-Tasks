# Aspect-Based Sentiment Analysis #
This repository contains the code and data for the "Czech Dataset for Complex Aspect-Based Sentiment Analysis Tasks" accepted to <a href="https://lrec-coling-2024.org/">LREC-COLING 2024</a> conference.

## Requirements ##
Python 3.10 and higher is required. The code was tested on Python 3.11.4.
The required Python packages are listed in the file `requirements.txt`. They can be installed using the command `pip install -r requirements.txt`.

## Structure ##
The repository contains the following files and folders:
* `main.py` – Main file of the application.
* `requirements.txt` – List of required Python packages.
* `README.md` – This file.
* `data` – Folder containing the data. See the section `Data` for more details.
* `src` – Folder containing the source code.
  * `data_utils` – Folder containing the code for data loading and preprocessing.
  * `evaluation` – Folder containing the code for evaluation.
  * `models` – Folder containing the code for models.
  * `utils` – Folder containing the code for other utilities.

## Data ##
There should be a folder `data` in the root of the repository. The folder should contain the following subfolders:
* `cs_rest_o` – Folder containing the data for CsRest-O dataset.
* `cs_rest_n` – Folder containing the data for CsRest-N dataset.
* `cs_rest_m` – Folder containing the data for CsRest-M dataset.

Each of the language folders should contain the following files:
* `train.xlm` – File with training data.
* `dev.xlm` – File with validation data.
* `test.xlm` – File with test data.

## Usage ##
To run the code, use the command `python main.py`. The code can be configured using the following command-line arguments:
* `--model` – Name or path to pre-trained model. The default value is `t5-base`.
* `--batch_size` – Batch size. The default value is `64`.
* `--max_seq_length` – Maximum sequence length. The default value is `256`.
* `--max_seq_length_label` – Maximum sequence length for labels. The default value is `256`.
* `--lr` – Learning rate. The default value is `1e-4`.
* `--epochs` – Number of training epochs. The default value is `10`.
* `--dataset_path` – Dataset path. The default value is `cs_rest_m`. Options: `cs_rest_o`, `cs_rest_n`, `cs_rest_m`.
* `--optimizer` – Optimizer. The default value is `AdamW`. Options: `adafactor`, `AdamW`. `adafactor` can be applied only for sequence-to-sequence models (e.g. T5), otherwise `AdamW` is always used.
* `--mode` – Mode of the training. The default value is `dev`. Options:
  * `dev` – Splits the training data into training and validation sets. The validation set is used for selecting the best model, which is then evaluated on the test set of the target language. In the case of different source and target languages, the validation set is taken from the training data of the target language and the whole training data is used for training.
  * `test` – Uses whole training dataset from the source language for training and test dataset from the target language for evaluation.
* `--checkpoint_monitor` – Metric based on which the best model will be stored according to the performance on validation data in `dev` mode. The default value is `val_loss`. Options: `val_loss`, `ate_f1`, `acd_f1`, `acte_f1`, `tasd_f1`, `pd_f1`, `e2e_f1`.
* `--accumulate_grad_batches` – Accumulates gradient batches. The default value is `1`. It is used when there is insufficient memory for training for the required effective batch size.
* `--beam_size` – Beam size for beam search decoding. The default value is `1`.
* `--task` – ABSA task. Options: `ate`, `e2e`, `apd`, `acd`.

### Restrictions and details ###
* `--task` cannot be used for sequence-to-sequence models (e.g. T5).
* When selecting a model for sentiment polarity classification (e.g. BERT), `--task` has to be used.
* The user is responsible for selecting the correct `--checkpoint_monitor` (e.g. `acd_f1` is only measured with `acd` task).
* The `-model` argument containing the substring `t5` or `bart` indicates the use of a sequence-to-sequence model.
* The program automatically detects whether it is possible to use GPU for training.
* The program tries to use CsvLogger.

### Examples ###
* `python main.py --model google/mt5-base --batch_size 32 --max_seq_length 256 --max_seq_length_label 256 --lr 3e-4 --epochs 35 --dataset_path cs_rest_o --optimizer adafactor --mode dev --checkpoint_monitor tasd_f1 --accumulate_grad_batches 2 --beam_size 1`
* `python main.py --model xlm-roberta-base --batch_size 64 --max_seq_length 256 --lr 1e-5 --epochs 50 --dataset_path cs_rest_o --optimizer AdamW --mode dev --checkpoint_monitor pd_f1 --accumulate_grad_batches 1 --task apd`
* `python main.py --model xlm-roberta-base --batch_size 64 --max_seq_length 256 --max_seq_length_label 256 --lr 1e-5 --epochs 50 --dataset_path cs_rest_m --optimizer AdamW --mode test --task ate`
* `python main.py --model xlm-roberta-base --batch_size 64 --max_seq_length 256 --max_seq_length_label 256 --lr 1e-5 --epochs 10 --dataset_path cs_rest_n --checkpoint_monitor acd_f1 --optimizer AdamW --mode dev --task acd`

## Citation
If you find this repository helpful for your research, please cite our paper as follows:
```
@inproceedings{smid-etal-2024-czech,
    title = "{C}zech Dataset for Complex Aspect-Based Sentiment Analysis Tasks",
    author = "{\v{S}}m{\'i}d, Jakub  and
      P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Prazak, Ondrej  and
      Kral, Pavel",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.384/",
    pages = "4299--4310",
    abstract = "In this paper, we introduce a novel Czech dataset for aspect-based sentiment analysis (ABSA), which consists of 3.1K manually annotated reviews from the restaurant domain. The dataset is built upon the older Czech dataset, which contained only separate labels for the basic ABSA tasks such as aspect term extraction or aspect polarity detection. Unlike its predecessor, our new dataset is specifically designed to allow its usage for more complex tasks, e.g. target-aspect-category detection. These advanced tasks require a unified annotation format, seamlessly linking sentiment elements (labels) together. Our dataset follows the format of the well-known SemEval-2016 datasets. This design choice allows effortless application and evaluation in cross-lingual scenarios, ultimately fostering cross-language comparisons with equivalent counterpart datasets in other languages. The annotation process engaged two trained annotators, yielding an impressive inter-annotator agreement rate of approximately 90{\%}. Additionally, we provide 24M reviews without annotations suitable for unsupervised learning. We present robust monolingual baseline results achieved with various Transformer-based models and insightful error analysis to supplement our contributions. Our code and dataset are freely available for non-commercial research purposes."
}
```
