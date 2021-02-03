import sys
# sys.path.append("../../")
sys.path.append("/home/cbarkhof/code-thesis/NewsVAE")
import train
import utils_train
import os
import validate_evaluate
from utils_evaluation import valid_dataset_loader_tokenizer
import pickle
from pathlib import Path
import time


def load_model_for_eval(path, device_name="cuda:0", latent_size=32, add_latent_via_memory=True,
                        add_latent_via_embeddings=False, do_tie_weights=True, do_tie_embedding_spaces=True,
                        add_decoder_output_embedding_bias=False):
    vae_model = train.get_model_on_device(device_name=device_name,
                                          latent_size=latent_size,
                                          gradient_checkpointing=False,
                                          add_latent_via_memory=add_latent_via_memory,
                                          add_latent_via_embeddings=add_latent_via_embeddings,
                                          do_tie_weights=do_tie_weights,
                                          do_tie_embedding_spaces=do_tie_embedding_spaces,
                                          world_master=True,
                                          add_decoder_output_embedding_bias=add_decoder_output_embedding_bias)

    _, _, vae_model, _, _, _, _ = utils_train.load_from_checkpoint(vae_model, path)
    vae_model.eval()
    return vae_model


def run_validation(max_batches=-1, batch_size_mi_calc=128, n_batches_mi_calc=20):
    dataset_path = "/home/cbarkhof/code-thesis/NewsVAE/NewsData/22DEC-cnn_dailymail-roberta-seqlen64/validation"

    valid_dataset, tokenizer, valid_loader = valid_dataset_loader_tokenizer(batch_size=64,
                                                                            num_workers=4,
                                                                            dataset_path=dataset_path)

    run_dir = Path('/home/cbarkhof/code-thesis/NewsVAE/Runs')

    run_names = []
    best_model_paths = []
    last_model_paths = []
    for run_name in os.listdir(run_dir):
        if ("29DEC" in run_name) or ("13JAN" in run_name) or ("12JAN" in run_name) or \
                ("7JAN" in run_name) or ("18NOV" in run_name) or ("23NOV-AUTOENCODER" in run_name):
            run_names.append(run_name)
            run_path = run_dir / run_name
            for c in os.listdir(run_path):
                if "checkpoint" in c:
                    if "best" in c:
                        best_model_paths.append(run_dir / run_name / c)
                    else:
                        last_model_paths.append(run_dir / run_name / c)

    print("Found the following run names relevant:")
    for i, (r, b, l) in enumerate(zip(run_names, best_model_paths, last_model_paths)):
        print(i, r)
        print(b)
        print(l)

    save_dir = Path("/home/cbarkhof/code-thesis/NewsVAE/evaluation/14JAN/results-validation/")
    os.makedirs(save_dir, exist_ok=True)

    for name, best_path, last_path in zip(run_names, best_model_paths, last_model_paths):
        print("-" * 100)
        print(f"Loading: {name}")

        if f"results_{name}.p" in os.listdir("/home/cbarkhof/code-thesis/NewsVAE/evaluation/14JAN/results-validation"):
            print("Did this one already, skipping it...")
            continue

        try:
            results_best_last = {}
            results_best_last["best_path"] = best_path
            results_best_last["last_path"] = last_path

            for path, best_last in zip([best_path, last_path], ["best", "last"]):

                # start timing
                start = time.time()

                # load model
                vae_model = load_model_for_eval(path, device_name="cuda:0")

                # run validation with model
                results_best_last[best_last] = validate_evaluate.validation_set_results(vae_model, valid_loader, tokenizer,
                                                                                        device_name="cuda:0",
                                                                                        max_batches=max_batches,
                                                                                        batch_size_mi_calc=batch_size_mi_calc,
                                                                                        n_batches_mi_calc=n_batches_mi_calc)


                # end timing
                end = time.time()
                hours, rem = divmod(end - start, 3600)
                minutes, seconds = divmod(rem, 60)

                # print timing
                print("Took: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            results_name = f"results_{name}.p"
            pickle.dump(results_best_last, open(save_dir / results_name, "wb"))
            print(f"Done with: {name}")

        except Exception as e:
            print("this model failed:", name)
            print("error", e)



if __name__ == "__main__":
    run_validation()
