import torch as torch
import transformers
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
#
# # TODO: get rid of this dangerous statement
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
transformers.logging.set_verbosity_error()

class NewsDataModule(pl.LightningDataModule):
    """
    A class to handle news data preparation, downloading and loading.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str,
                 batch_size: int = 8, num_workers: int = 4,
                 pin_memory: bool = True):
        super().__init__()
        self.pin_memory = pin_memory

        # DATASET PROPERTIES
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.num_workers = num_workers

        self.datasets = {}

        # TOKENIZER
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def setup(self, stage: Optional[str] = None):
        name = '3.0.0' if self.dataset_name == 'cnn_dailymail' else None
        assert self.dataset_name in list_datasets(), "Currently only supporting datasets from Huggingface"

        for split in ['train', 'validation', 'test']:
            self.datasets[split] = load_dataset(self.dataset_name, name=name, ignore_verifications=True,
                                                split=split+'[:100]')
            self.datasets[split] = self.datasets[split].map(self.convert_to_features, batched=True)
            columns = ['attention_mask', 'input_ids']

            self.datasets[split].set_format(type='torch', columns=columns)

    # def transfer_batch_to_device(self, batch, device):
    #     for k in batch.keys():
    #         batch[k] = batch[k].to(device)
    #     return batch

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.datasets['train'], collate_fn=self.collate_fn,
                                  batch_size=self.batch_size, num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self.datasets['validation'], collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory)
        return val_loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_loader = DataLoader(self.datasets['test'], collate_fn=self.collate_fn,
                                 batch_size=self.batch_size, num_workers=self.num_workers,
                                 pin_memory=self.pin_memory)
        return test_loader

    def collate_fn(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        A function that assembles a batch. This is where padding is done, since it depends on
        the maximum sequence length in the batch.

        :param examples: list of truncated, tokenised & encoded sequences
        :return: padded_batch (batch x max_seq_len)
        """

        print("-> collate_fn")

        # Combine the tensors into a padded batch
        padded_batch: Dict[str, torch.Tensor] = self.tokenizer.pad(examples, return_tensors='pt')

        return padded_batch

    def convert_to_features(self, data_batch: OrderedDict) -> OrderedDict:
        """
        Truncates and tokenises & encodes a batch of text samples.

        ->  Note: does not pad yet, this will be done in the DataLoader to allow flexible
            padding according to the longest sequence in the batch.

        :param data_batch: batch of text samples
        :return: encoded_batch: batch of samples with the encodings with the defined tokenizer added
        """

        print("-> convert_to_features")

        encoded_batch = self.tokenizer(data_batch['article'], truncation=True)

        return encoded_batch


if __name__ == "__main__":
    data = NewsDataModule('cnn_dailymail', 'roberta')
    data.setup()
    data.prepare_data()

    for batch in data.train_dataloader():
        for k, v in batch.items():
            print(k)
            print(v.shape)
        break
