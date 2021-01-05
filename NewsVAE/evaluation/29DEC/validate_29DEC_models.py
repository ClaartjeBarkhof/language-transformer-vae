import train
import utils_train
import os
import sys; sys.path.append("../../")
import validate_evaluate
from utils_evaluation import valid_dataset_loader_tokenizer
import pickle


def load_model_for_eval(path, device_name="cuda:0"):
    vae_model = train.get_model_on_device(device_name=device_name,
                                          latent_size=768,
                                          gradient_checkpointing=False,
                                          add_latent_via_memory=True,
                                          add_latent_via_embeddings=True,
                                          do_tie_weights=True,
                                          world_master=True)

    _, _, vae_model, _, _, _, _ = utils_train.load_from_checkpoint(vae_model, path)
    vae_model.eval()
    return vae_model


def validate_models_29dec(max_batches=-1, batch_size_mi_calc=128, n_batches_mi_calc=20):
    dataset_path = "/home/cbarkhof/code-thesis/NewsVAE/NewsData/22DEC-cnn_dailymail-roberta-seqlen64/validation"
    valid_dataset, tokenizer, valid_loader = valid_dataset_loader_tokenizer(batch_size=64,
                                                                            num_workers=4,
                                                                            dataset_path=dataset_path)

    run_dir = '/home/cbarkhof/code-thesis/NewsVAE/Runs'

    runs_29DEC_names = ["-".join(run_name.split('-')[2:-5]) for run_name in os.listdir(run_dir) if "29DEC" in run_name]
    runs_29DEC_paths = [run_dir + '/' + run_name + '/checkpoint-best.pth' for run_name in os.listdir(run_dir) if
                        "29DEC" in run_name]

    save_dir = "/home/cbarkhof/code-thesis/NewsVAE/evaluation/29DEC/results-validation/"
    os.makedirs(save_dir, exist_ok=True)

    for name, path in zip(runs_29DEC_names, runs_29DEC_paths):
        print("-" * 100)
        print(f"Loading: {name}")
        vae_model = load_model_for_eval(path, device_name="cuda:0")
        results = validate_evaluate.validation_set_results(vae_model, valid_loader, tokenizer, device_name="cuda:0",
                                                           max_batches=max_batches, batch_size_mi_calc=batch_size_mi_calc,
                                                           n_batches_mi_calc=n_batches_mi_calc)
        pickle.dump(results, open(save_dir + "results_" + name + ".p", "wb"))
        print(f"Done with: {name}")


if __name__ == "__main__":
    validate_models_29dec()