# Basic imports
import os
import numpy as np
import torch
import pickle
import pandas as pd
import re

# Plotting
# %matplotlib inline
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Globals
RUN_DIR = "/home/cbarkhof/code-thesis/NewsVAE/Runs/"
RES_FILE_DIR = "/home/cbarkhof/code-thesis/NewsVAE/final-analysis/result-files"
OVERVIEW_CSV = "/home/cbarkhof/code-thesis/NewsVAE/final-analysis/run_overview.csv"


def get_clean_name(run_name, overview_df=None, rerun_overview_csv=False, overview_csv_file=None):
    if overview_csv_file is None:
        overview_csv_file = OVERVIEW_CSV

    if overview_df is None:
        if rerun_overview_csv:
            make_run_overview_csv(overview_csv_file=overview_csv_file)
        overview_df = pd.read_csv(overview_csv_file, index_col=0)

    clean_name = overview_df[overview_df["run_name"] == run_name]["clean_name"].values[0]
    return clean_name


def read_overview_csv(overview_csv_file=None):
    if overview_csv_file is None:
        overview_csv_file = OVERVIEW_CSV

    make_run_overview_csv(overview_csv_file=overview_csv_file)

    return pd.read_csv(overview_csv_file, index_col=0)


def get_run_name(clean_name, overview_df=None, rerun_overview_csv=False, overview_csv_file=None):
    if overview_csv_file is None:
        overview_csv_file = OVERVIEW_CSV

    # If no df passed on
    if overview_df is None:
        if rerun_overview_csv:
            make_run_overview_csv(overview_csv_file=overview_csv_file)
        overview_df = pd.read_csv(overview_csv_file, index_col=0)

    run_name = overview_df[overview_df["clean_name"] == clean_name]["run_name"].values[0]

    return run_name


def get_last_pareto_dict(run_name):
    result_pickle = f"{RES_FILE_DIR}/{run_name}/last_pareto_dict.p"
    if os.path.isfile(result_pickle):
        pareto_dict = pickle.load(open(result_pickle, "rb"))
        return pareto_dict
    else:
        print("No pareto dict present, perhaps re-run <save_last_pareto_dict_to_pickle> "
              "to unpack latests data from checkpoints.")


def get_best_checkpoint(run_name):
    # Get best epoch
    if os.path.isdir(f"{RES_FILE_DIR}/{run_name}"):
        for f in os.listdir(f"{RES_FILE_DIR}/{run_name}"):
            if "weighted_pareto" in f:
                best_epoch = int(re.split('\[|\]', f)[-2])

        path = None
        # Get checkpoint belonging to that epoch
        for f in os.listdir(RUN_DIR + run_name):
            if f"epoch-{best_epoch:03d}" in f:
                path = RUN_DIR + run_name + "/" + f

        return path, best_epoch
    else:
        print("no weighted pareto scores for this run yet", run_name)
        return None, None


def parse_run_name(run_name):
    # Dataset name
    if "PTB" in run_name:
        dataset = "PTB"
    elif "OPTIMUS" in run_name:
        dataset = "OPTIMUS YELP"
    else:
        dataset = "YELP"

    # Optimisation
    if "VAE" in run_name:
        opt = "VAE"
        tr = 0.0
    elif "CYC" in run_name:
        opt = "CYC-FB-0.5"
        tr = 0.5
    else:
        opt = "MDR-0.5"
        tr = 0.5

    # Decoder drop-out
    if "DROP 40" in run_name:
        drop_str = " | DROP 40"
        drop = 0.4
    else:
        drop_str = ""
        drop = 0.0

    # Latent mechanism
    matrix, emb, mem = False, False, False
    if "matrix" in run_name:
        matrix = True
    if "memory" in run_name:
        mem = True
    if "emb" in run_name:
        emb = True

    mech_string_list = []
    if matrix:
        mech_string_list.append("matrix")
    if mem:
        mech_string_list.append("mem")
    if emb:
        mech_string_list.append("emb")

    mech_string = "-".join(mech_string_list)

    clean_name = f"{dataset} | {opt} | {mech_string}{drop_str}"

    parsed_run_name = {
        "run_name": run_name,
        "clean_name": clean_name,
        "dataset": dataset,
        "optimisation": opt,
        "drop": drop,
        "target_rate": tr,
        "matrix": matrix,
        "emb": emb,
        "mem": mem}

    return parsed_run_name


def make_run_overview_csv(overview_csv_file=None):
    if overview_csv_file is None:
        overview_csv_file = OVERVIEW_CSV

    overview_rows = []
    for run_name in os.listdir(RUN_DIR):

        row = parse_run_name(run_name)

        overview_rows.append(row)

    run_overview = pd.DataFrame(overview_rows)
    print(f"Saving overview in {overview_csv_file}")
    run_overview.to_csv(overview_csv_file)


def save_last_pareto_dict_to_pickle():
    for run_name in os.listdir(RUN_DIR):
        try:
            # Check if already done
            result_pickle = f"{RES_FILE_DIR}/{run_name}/last_pareto_dict.p"
            if os.path.isfile(result_pickle):
                continue

            # Get the last checkpoint saved
            checkpoint_list = sorted([c for c in os.listdir(RUN_DIR + run_name) if c != "wandb"])
            last_checkpoint = checkpoint_list[-1]

            path = RUN_DIR + run_name + f"/{last_checkpoint}"

            # Load the checkpoint
            c = torch.load(path)
            last_pareto_dict = c["epoch_pareto_effiency_dict"]

            # Make a folder for the run name if it does not exist already
            os.makedirs(f"result-files/{run_name}", exist_ok=True)

            # Dump the pareto dict
            pickle.dump(last_pareto_dict, open(result_pickle, "wb"))
        except Exception as e:
            print(run_name)
            print("error:", e)


def get_pareto_efficient_dict(run_name):
    pareto_pickle = f"result-files/{run_name}/last_pareto_dict.p"

    if not os.path.isfile(pareto_pickle):
        print("Could not find pareto pickle, re-unpacking pareto dicts from checkpoints.")
        save_last_pareto_dict_to_pickle()

    pareto_dict = pickle.load(open(pareto_pickle, "rb"))
    pareto_eff_dict = {"efficient_epochs": pareto_dict["efficient_epochs"]}

    for i, name in enumerate(["iw_ll_mean", "iw_ll_x_gen_mean", "-D_ks"]):
        pareto_eff_dict[name] = []
        for e in pareto_dict["efficient_epochs"]:
             pareto_eff_dict[name].append(pareto_dict[name][e])

    return pareto_eff_dict


def plot_pareto_stats(run_name, clean_name=None, best_epoch=None):
    pareto_pickle = f"result-files/{run_name}/last_pareto_dict.p"

    if not os.path.isfile(pareto_pickle):
        print("Could not find pareto pickle, re-unpacking pareto dicts from checkpoints.")
        save_last_pareto_dict_to_pickle()

    pareto_dict = pickle.load(open(pareto_pickle, "rb"))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    if clean_name is None:
        clean_name = get_clean_name(run_name)

    fig.suptitle(clean_name, y=1.05, size=12)

    for i, name in enumerate(["iw_ll_mean", "iw_ll_x_gen_mean", "-D_ks"]):
        axs[i].plot(pareto_dict[name])
        axs[i].set_title(name)

        for e in pareto_dict["efficient_epochs"]:
            axs[i].scatter(e, pareto_dict[name][e], color='r')

        if best_epoch is not None:
            axs[i].scatter(best_epoch, pareto_dict[name][best_epoch], color='g')

    plt.show()