from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
from torch.multiprocessing import Pool, Process, set_start_method

benchmark_args = PyTorchBenchmarkArguments(models=["bert-base-uncased"],
                                           batch_sizes=[8],
                                           sequence_lengths=[8, 32, 128, 512],
                                           save_to_csv=True,
                                           log_filename='log',
                                           env_info_csv_file='env_info')

benchmark = PyTorchBenchmark(benchmark_args)
benchmark.run()
