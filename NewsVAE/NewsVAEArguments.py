import argparse


def preprare_parser():
    parser = argparse.ArgumentParser()

    # DATA & TOKENISATION
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--dataset_name", default='cnn_dailymail', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default.")

    # TRAIN
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Num workers for data loading.")

    # LOSS
    parser.add_argument("--KL_mask_value", default=3.0, type=float,
                        help="The ....")  # TODO: check this and finish this

    args = parser.parse_args()

    return args
