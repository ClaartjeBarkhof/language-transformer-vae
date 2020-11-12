from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import NewsVAEArguments
from transformers import RobertaTokenizer
import torch
from NewsData import NewsData
import numpy as np
from functools import partial
from datasets import load_dataset

def get_random_batch_ids(low, high, size):
    random_ints = []
    while len(random_ints) != size:
        r = np.random.randint(low, high)
        if r not in random_ints:
            random_ints.append(r)
    return list(random_ints)

def get_evaliation_random_batch(dataset, batchsize=32):
    ids = get_random_batch_ids(0, len(dataset), batchsize)
    attention_mask_batch = torch.LongTensor(dataset[ids]['attention_mask'])
    input_ids_batch = torch.LongTensor(dataset[ids]['input_ids'])
    input_batch = {'attention_mask': attention_mask_batch, 'input_ids':input_ids_batch}
    return dataset[ids], input_batch


def convert_to_features(data_batch, tokenizer=None, args=None):
    encoded_batch = tokenizer(data_batch['article'], truncation=True, max_length=args.max_seq_len, padding=True)
    encoded_batch['article'] = [a for a in data_batch['article']]
    encoded_batch['id'] = [i for i in data_batch['id']]
    return encoded_batch

def make_evaluation_dataset(validation_dataset):



def evaluate(VAE_model, validation_dataset, wandb_object, num_evaluation_samples, tokenizer, args):
    encoded_dataset_validation = validation_dataset.map(partial(convert_to_features, tokenizer=tokenizer, args=args), batched=True)