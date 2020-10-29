import torch as torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, logging as transformers_logging  # type: ignore
from datasets import load_dataset, list_datasets, load_from_disk  # type: ignore
from typing import List, Dict, Union, Optional
from collections import OrderedDict
import os
import pytorch_lightning as pl
import utils
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO: get rid of this dangerous statement
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

transformers_logging.set_verbosity_warning()


class NewsDataModule(pl.LightningDataModule):
    """
    A class to handle news data preparation, downloading and loading.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str,
                 batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.DATASETS_PROPERTIES = {
            'cnn_dailymail': {
                'article_col': 'article',
                'splits': ['train', 'validation', 'test'],
            },
            'ag_news': {
                'article_col': 'text',
                'splits': ['train', 'test']
            }
        }

        self.TOKENIZER_PROPERTIES = {
            'roberta': {
                'class': RobertaTokenizerFast,
                'ckpt': 'roberta-base'
            }
        }

        # DATASET PROPERTIES
        self.dataset_name = dataset_name
        self.splits = self.DATASETS_PROPERTIES[dataset_name]['splits']
        self.article_column = self.DATASETS_PROPERTIES[dataset_name]['article_col']
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.num_workers = num_workers

        # TOKENIZER
        self.tokenizer = self.TOKENIZER_PROPERTIES[self.tokenizer_name]['class'].from_pretrained(
            self.TOKENIZER_PROPERTIES[self.tokenizer_name]['ckpt'])

        print("check check")

    def setup(self, stage: Optional[str] = None):
        pass

    def transfer_batch_to_device(self, batch, device):
        print("### DEVICE CHECK", device)
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        return batch

    def prepare_data(self):
        # ENCODE DATASET PATHS
        file_encoded_dataset = '/{}-{}'.format(self.dataset_name, self.tokenizer_name)
        data_dir = utils.get_code_dir() + 'Data'
        file_path_encoded_dataset = data_dir + file_encoded_dataset

        # LOAD PROCESSED FROM DISK
        if os.path.isdir(file_path_encoded_dataset):
            self.dataset = load_from_disk(data_dir + file_encoded_dataset)

        # LOAD & PROCESS DATA
        else:
            name = '3.0.0' if self.dataset_name == 'cnn_dailymail' else None
            assert self.dataset_name in list_datasets(), "Currently only supporting datasets from Huggingface"
            self.dataset = load_dataset(self.dataset_name, name=name, ignore_verifications=True)
            self.change_article_col_name()

            self.dataset = self.dataset.map(self.convert_to_features, batched=True)
            # Not the article itself since it cant be turned into torch tensor, and will break up the dataloading
            columns = ['attention_mask', 'input_ids']
            self.dataset.set_format(type='torch', columns=columns)
            print("Saving processed dataset to disk: {}".format(file_path_encoded_dataset))
            self.dataset.save_to_disk(file_path_encoded_dataset)

    def train_dataloader(self) -> DataLoader:
        print("trainloader")
        train_loader = DataLoader(self.dataset['train'], collate_fn=self.collate_fn,
                                  batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        print("valloader")
        val_loader = DataLoader(self.dataset['validation'], collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        print("testloader")
        test_loader = DataLoader(self.dataset['test'], collate_fn=self.collate_fn,
                                 batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader

    def collate_fn(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        A function that assembles a batch. This is where padding is done, since it depends on
        the maximum sequence length in the batch.

        :param examples: list of truncated, tokenised & encoded sequences
        :return: padded_batch (batch x max_seq_len)
        """

        # Combine the tensors into a padded batch
        padded_batch: Dict[str, torch.Tensor] = self.tokenizer.pad(examples, return_tensors='pt')

        return padded_batch

    def change_article_col_name(self) -> None:
        """
        Changes the article text column name to 'article' for consistency.
        """
        article_col_name = self.DATASETS_PROPERTIES[self.dataset_name]['article_col']
        if article_col_name != 'article':
            self.dataset = self.dataset.map(lambda example: {'article': example[article_col_name]},
                                            remove_columns=[article_col_name])

    def convert_to_features(self, data_batch: OrderedDict) -> OrderedDict:
        """
        Truncates and tokenises & encodes a batch of text samples.

        ->  Note: does not pad yet, this will be done in the DataLoader to allow flexible
            padding according to the longest sequence in the batch.

        :param data_batch: batch of text samples
        :return: encoded_batch: batch of samples with the encodings with the defined tokenizer added
        """
        encoded_batch = self.tokenizer(data_batch['article'], truncation=True)

        return encoded_batch


if __name__ == "__main__":
    data = NewsDataModule('cnn_dailymail', 'roberta')
    data.setup()
    data.prepare_data()