import torch as torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, logging as transformers_logging  # type: ignore
from datasets import load_dataset, list_datasets, load_from_disk # type: ignore
from typing import Optional, List, Dict, Union
import os
from collections import OrderedDict
import os

# TODO: get rid of this dangerous statement
os.environ['KMP_DUPLICATE_LIB_OK']='True'

transformers_logging.set_verbosity_warning()

DATASETS_PROPERTIES = {
    'cnn_dailymail': {
        'article_col': 'article',
        'splits': ['train', 'validation', 'test'],
    },
    'ag_news': {
        'article_col': 'text',
        'splits': ['train', 'test']
    }
}

TOKENIZER_PROPERTIES = {
    'roberta': {
        'class': RobertaTokenizerFast,
        'ckpt': 'roberta-base'
    }
}


class NewsData:
    """
    A class to handle news data preparation, downloading and loading.
    """
    def __init__(self, dataset_name: str, tokenizer_name: str,
                 batch_size: int = 8, num_workers: int = 8):
        # DATASET PROPERTIES
        self.dataset_name = dataset_name
        self.splits = DATASETS_PROPERTIES[dataset_name]['splits']
        self.article_column = DATASETS_PROPERTIES[dataset_name]['article_col']

        # TOKENIZER
        self.tokenizer = TOKENIZER_PROPERTIES[tokenizer_name]['class'].from_pretrained(
            TOKENIZER_PROPERTIES[tokenizer_name]['ckpt'])

        # ENCODE DATASET PATHS
        file_encoded_dataset = '{}-{}'.format(dataset_name, tokenizer_name)
        data_dir = '../Data/'
        file_path_encoded_dataset = data_dir + file_encoded_dataset

        # LOAD PROCESSED FROM DISK
        if file_encoded_dataset in os.listdir(data_dir):
            print("Encoded this one before, loading from disk: {}".format(file_path_encoded_dataset))
            self.dataset = load_from_disk(data_dir+file_encoded_dataset)

        # LOAD & PROCESS DATA
        else:
            print("Did not encode this one before, loading and processing...")
            name = '3.0.0' if dataset_name == 'cnn_dailymail' else None
            assert self.dataset_name in list_datasets(), "Currently only supporting datasets from Huggingface"
            self.dataset = load_dataset(dataset_name, name=name)
            self.change_article_col_name()

            self.dataset = self.dataset.map(self.convert_to_features, batched=True)
            print(self.dataset.column_names)
            # Not the article itself since it cant be turned into torch tensor
            columns = ['attention_mask', 'input_ids']
            self.dataset.set_format(type='torch', columns=columns)
            print("Saving processed dataset to disk: {}".format(file_path_encoded_dataset))
            self.dataset.save_to_disk(file_path_encoded_dataset)

        # PREPARE DATA LOADERS
        self.dataloaders = {split: DataLoader(self.dataset[split],
                                              collate_fn=self.collate_fn,
                                              batch_size=batch_size,
                                              num_workers=num_workers) for split in self.splits}

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
        article_col_name = DATASETS_PROPERTIES[self.dataset_name]['article_col']
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
        # TODO: check whether this adds start and end of sentence tokens
        encoded_batch = self.tokenizer(data_batch['article'], truncation=True)

        return encoded_batch


if __name__ == "__main__":
    data = NewsData('cnn_dailymail', 'roberta')

    for batch_idx, batch in enumerate(data.dataloaders['train']):
        print("Batch {}".format(batch_idx))
        print("Batch consists of {}".format(batch.keys()))
        print("Of shapes {}".format([batch[key].shape for key in batch.keys()]))
        break
