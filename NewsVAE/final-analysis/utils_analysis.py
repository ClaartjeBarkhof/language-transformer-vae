# Basic imports
import os
import numpy as np
import torch
import pickle
import pandas as pd
import re
import sys; sys.path.append("../")

# Plotting
# %matplotlib inline
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

# Globals
RUN_DIRS = {
    "Runs": "/home/cbarkhof/code-thesis/NewsVAE/Runs",
    "Runs-ablation": "/home/cbarkhof/code-thesis/NewsVAE/Runs-ablation",
    "Runs-target-rate": "/home/cbarkhof/code-thesis/NewsVAE/Runs-target-rate"
}

RES_FILE_DIR = "/home/cbarkhof/code-thesis/NewsVAE/final-analysis/result-files"
ANALYSIS_DIR = "/home/cbarkhof/code-thesis/NewsVAE/final-analysis"


def update():
    for exp_name, run_dir in RUN_DIRS.items():

        print(f"Making run overview, based on dir: {run_dir}")
        run_overview = make_run_overview_csv(run_dir, exp_name)

        print("Reading last checkpoint and extracting pareto dict and saving it to a pickle.")
        save_last_pareto_dict_to_pickle(run_overview, run_dir, exp_name)

        print("Reading all pareto dicts and calculating best checkpoint, saving it to a csv")
        calc_weighted_pareto_best_checkpoint(run_overview, exp_name)

        print("-" * 50)


def make_run_overview_csv(run_dir, exp_name):
    overview_csv_file = f"{ANALYSIS_DIR}/{exp_name}_run_overview.csv"

    print(f"Making run overview of {run_dir}, in {overview_csv_file}")

    overview_rows = []
    for run_name in os.listdir(run_dir):
        row = parse_run_name(run_name, exp_name)
        overview_rows.append(row)

    run_overview = pd.DataFrame(overview_rows)
    run_overview.to_csv(overview_csv_file)

    return run_overview


def save_last_pareto_dict_to_pickle(run_overview, run_dir, exp_name):
    for row_i, row in run_overview.iterrows():
        run_name = row["run_name"]
        try:
            # Check if already done
            result_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"

            if os.path.isfile(result_pickle): # uncomment this when everything is done running
                continue

            # Get the last checkpoint saved
            checkpoint_list = sorted([c for c in os.listdir(f"{run_dir}/{run_name}") if c != "wandb"])
            last_checkpoint = checkpoint_list[-1]

            path = f"{run_dir}/{run_name}/{last_checkpoint}"

            # Load the checkpoint
            c = torch.load(path)
            last_pareto_dict = c["epoch_pareto_effiency_dict"]

            # Make a folder for the run name if it does not exist already
            os.makedirs(f"result-files/{exp_name}/{run_name}", exist_ok=True)

            # Dump the pareto dict
            pickle.dump(last_pareto_dict, open(result_pickle, "wb"))

        except Exception as e:
            print(run_name)
            print("error:", e)


def parse_run_name(run_name, exp_name):
    parsed_run_name = None

    if exp_name == "Runs":
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

        mech_string = "+".join(mech_string_list)

        clean_name = f"{dataset} | {opt} | {mech_string}{drop_str}"

        parsed_run_name = {
            "run_name": run_name,
            "clean_name": clean_name,
            "mech_string": mech_string,
            "dataset": dataset,
            "optimisation": opt,
            "drop": drop,
            "target_rate": tr,
            "matrix": matrix,
            "emb": emb,
            "mem": mem}

    elif exp_name == "Runs-target-rate":
        target_rate = run_name.split("-")[6]
        clean_name = f"PTB | FB | mem+emb | Target rate: {target_rate}"

        parsed_run_name = {
            "run_name": run_name,
            "mech_string": "mem+emb",
            "dataset_name": "PTB",
            "dataset": "PTB",
            "clean_name": clean_name,
            "target_rate": target_rate,
            "optimisation":"FB-0.5",
            "drop":0.0,
            "mem": True,
            "emb": True,
            "matrix": False
        }

    elif exp_name == "Runs-ablation":
        opt_list = []
        if "cyclical" in run_name:
            opt_list.append("CYC")
        if "free" in run_name:
            opt_list.append("FB-0.5")
        if "mdr" in run_name:
            opt_list.append("MDR-0.5")
        opt_string = "+".join(opt_list)
        clean_name = f"YELP | {opt_string} | mem"

        parsed_run_name = {
            "clean_name": clean_name,
            "run_name": run_name,
            "drop":0.0,
            "dataset": "YELP",
            "optimisation": opt_string,
            "mem": True,
            "emb": False,
            "matrix": False
        }

    else:
        print(f"{exp_name} is not a valid experiment name, or experiment folder.")

    return parsed_run_name


def remove_least_efficient_checkpoints(exp_name="Runs", max_n_checkpoints=5):
    df = read_overview_csv(exp_name)

    for row_i, row in df.iterrows():
        run_name = row["run_name"]
        print(row["clean_name"])

        least_efficient_epochs = []
        for f in os.listdir(f"result-files/{exp_name}/{run_name}"):
            if "weighted" in f:
                mini_df = pd.read_csv(f"result-files/{exp_name}/{run_name}/{f}", index_col=0)
                #display(mini_df)

                if len(mini_df) > max_n_checkpoints:
                    least_efficient_epochs = mini_df.iloc[5:]["efficient_epochs"].values
                    print("Least efficient epochs:", least_efficient_epochs)
                else:
                    print("Nothing to delete (less than 5 checkpoints).")

        run_dir = RUN_DIRS[exp_name]
        epoch_files = {}
        for f in os.listdir(f"{run_dir}/{run_name}"):
            if not "wandb" in f:
                epoch = int(f.split("-")[2])
                if epoch in least_efficient_epochs:
                    p = f"{run_dir}/{run_name}/{f}"
                    if os.path.exists(p):
                        print("Removing:", f)
                        os.remove(p)


def get_sum_stats_run(run_name, exp_name="Runs", val_batches=10):
    return pickle.load(open(f"{RES_FILE_DIR}/{exp_name}/{run_name}/validation_results_{val_batches}_batches.p", "rb"))


def calc_runs_missing(run_overview=None):
    if run_overview is None:
        run_overview = read_overview_csv()

    big_exp = {}
    i = 0
    for dataset in ["YELP", "OPTIMUS YELP", "PTB"]:
        for drop_out in [0.0, 0.4]:
            drop_str = " | DROP 40" if drop_out == 0.4 else ""
            for optimisation in ["CYC-FB-0.5", "MDR-0.5", "VAE"]:
                for mech in ["matrix", "matrix-mem", "mem", "mem-emb"]:
                    big_exp[i] = {
                        "optimisation": optimisation,
                        "drop_out": drop_out,
                        "mech": mech,
                        "dataset": dataset,
                        "clean_name": f"{dataset} | {optimisation} | {mech}{drop_str}"
                    }
                    i += 1

    all_exp = pd.DataFrame(big_exp).transpose()
    i = 0
    print("Runs missing in big experiment:\n")
    for clean_name in all_exp["clean_name"].values:
        if clean_name not in run_overview["clean_name"].values:
            print(i, clean_name)
            i += 1


def min_max_scaling(df, pareto_stats, scale_affix):
    # apply min-max scaling
    for column in pareto_stats:
        df[column + scale_affix] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


def calc_weighted_pareto_best_checkpoint(run_overview, exp_name):
    pareto_stats_weights = {
        "iw_ll_mean": 4,
        "iw_ll_x_gen_mean": 3,
        "-D_ks": 1
    }

    scale_affix = "_minmax_norm"

    for row_i, row in run_overview.iterrows():
        run_name = row["run_name"]

        try:

            for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}/"):
                if "weighted_score" in f:
                    continue

            mini_df = pd.DataFrame(get_pareto_efficient_dict(row["run_name"], exp_name=exp_name))

            if len(mini_df) == 1:
                for column in pareto_stats_weights.keys():
                    mini_df[column + scale_affix] = 1.0
            else:
                mini_df = min_max_scaling(mini_df, list(pareto_stats_weights.keys()), scale_affix)

            mini_df["combined_weighted_score"] = 0.0
            for k, w in pareto_stats_weights.items():
                mini_df["combined_weighted_score"] += w * mini_df[k + scale_affix]

            mini_df = mini_df.sort_values("combined_weighted_score", ascending=False)

            best_epoch = \
                mini_df[mini_df["combined_weighted_score"] == mini_df["combined_weighted_score"].max()][
                    "efficient_epochs"].values[
                    0]

            path = f"{RES_FILE_DIR}/{exp_name}/{run_name}/weighted_pareto_optimal_point_best[{best_epoch}].csv"

            mini_df.to_csv(path)

        except Exception as e:
            print("error in calc_weighted_pareto_best_checkpoint", e)


def read_overview_csv(exp_name="Runs"):
    overview_csv_file = f"{ANALYSIS_DIR}/{exp_name}_run_overview.csv"

    return pd.read_csv(overview_csv_file, index_col=0)


def get_last_pareto_dict(run_name, exp_name="Runs"):
    result_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"

    if os.path.isfile(result_pickle):
        pareto_dict = pickle.load(open(result_pickle, "rb"))
        return pareto_dict
    else:
        print("No pareto dict present, perhaps re-run <save_last_pareto_dict_to_pickle> "
              "to unpack latests data from checkpoints.")


def get_best_checkpoint(run_name, exp_name="Runs"):
    run_dir = RUN_DIRS[exp_name]

    # Get best epoch
    if os.path.isdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
        for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
            if "weighted_pareto" in f:
                best_epoch = int(re.split('\[|\]', f)[-2])

        path = None

        # Get checkpoint belonging to that epoch
        for f in os.listdir(f"{run_dir}/{run_name}"):
            if f"epoch-{best_epoch:03d}" in f:
                path = f"{run_dir}/{run_name}/{f}"

        return path, best_epoch

    else:
        print("no weighted pareto scores for this run yet", run_name)
        return None, None


def get_pareto_efficient_dict(run_name, exp_name="Runs"):
    pareto_pickle = f"result-files/{exp_name}/{run_name}/last_pareto_dict.p"

    if not os.path.isfile(pareto_pickle):
        print("get_pareto_efficient_dict: Could not find pareto pickle, try running update().")
        return None

    pareto_dict = pickle.load(open(pareto_pickle, "rb"))
    pareto_eff_dict = {"efficient_epochs": pareto_dict["efficient_epochs"]}

    for i, name in enumerate(["iw_ll_mean", "iw_ll_x_gen_mean", "-D_ks"]):
        pareto_eff_dict[name] = []
        for e in pareto_dict["efficient_epochs"]:
            pareto_eff_dict[name].append(pareto_dict[name][e])

    return pareto_eff_dict


def plot_pareto_stats(run_name, clean_name=None, best_epoch=None, exp_name="Runs"):
    pareto_pickle = f"result-files/{exp_name}/{run_name}/last_pareto_dict.p"

    if not os.path.isfile(pareto_pickle):
        print("plot_pareto_stats: Could not find pareto pickle, try running update().")
        return None

    if best_epoch is None:
        _, best_epoch = get_best_checkpoint(run_name=run_name, exp_name=exp_name)

    pareto_dict = pickle.load(open(pareto_pickle, "rb"))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    if clean_name is None:
        clean_name = run_name

    fig.suptitle(clean_name, y=1.05, size=12)

    for i, name in enumerate(["iw_ll_mean", "iw_ll_x_gen_mean", "-D_ks"]):
        axs[i].plot(pareto_dict[name])
        axs[i].set_title(name)

        for e in pareto_dict["efficient_epochs"]:
            axs[i].scatter(e, pareto_dict[name][e], color='r')

        if best_epoch is not None:
            axs[i].scatter(best_epoch, pareto_dict[name][best_epoch], color='g')

    plt.show()


update()
