from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments

# models = ["bert-base-cased", "bert-base-uncased", "gpt2",
#           "distilbert-base-cased", "distilbert-base-uncased",
#           "roberta-base", "google/roberta2roberta_L-24_cnn_daily_mail",
#           "patrickvonplaten/bert2bert-cnn_dailymail-fp16",
#           "WikinewsSum/bert2bert-multi-en-wiki-news"]
#
# sequence_lengths = [64, 128, 512]
# batch_sizes = [64, 128, 512]

benchmark_args = PyTorchBenchmarkArguments(models=["bert-base-uncased"],
                                           batch_sizes=[8],
                                           sequence_lengths=[8, 32, 128, 512],
                                           save_to_csv=True,
                                           log_filename='log',
                                           env_info_csv_file='env_info')
benchmark = PyTorchBenchmark(benchmark_args)
benchmark.run()