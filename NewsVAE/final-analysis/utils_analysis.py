# Basic imports
import os
import numpy as np
import torch
import pickle
import pandas as pd
import re
import sys;

sys.path.append("../")
from utils_evaluation import get_wandb_run_id, dump_pickle, load_pickle
import wandb

# Plotting
# %matplotlib inline
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

if os.path.exists("/home/cbarkhof"):
    user = "cbarkhof"
else:
    user = "ec2-user"

# Globals
RUN_DIRS = {
    "Runs": f"/home/{user}/code-thesis/NewsVAE/Runs",
    "Runs-ablation": f"/home/{user}/code-thesis/NewsVAE/Runs-ablation",
    "Runs-target-rate": f"/home/{user}/code-thesis/NewsVAE/Runs-target-rate",
    "Runs-pretrain": f"/home/{user}/code-thesis/NewsVAE/Runs-pretrain",
}

RES_FILE_DIR = f"/home/{user}/code-thesis/NewsVAE/final-analysis/result-files"
ANALYSIS_DIR = f"/home/{user}/code-thesis/NewsVAE/final-analysis"


def update(recompute=False):
    if recompute:
        print("Deleting all parato related files and recomputing.")
        delete_all_pareto_related_files()

    else:
        print("Not deleting all pareto related files, if you want to recompute, run: update(recompute=True)")

    for exp_name, run_dir in RUN_DIRS.items():

        if os.path.exists(run_dir):

            print(f"Making run overview, based on dir: {run_dir}")
            run_overview = make_run_overview_csv(run_dir, exp_name)

            print("Reading last checkpoint and extracting pareto dict and saving it to a pickle.")
            save_last_pareto_dict_to_pickle(run_overview, run_dir, exp_name)
            save_full_wandb_pareto_dict_to_pickle(run_overview, exp_name)

            print("Reading all pareto dicts and calculating best checkpoint, saving it to a csv")
            calc_weighted_pareto_best_checkpoint_all(run_overview, exp_name)

            print("-" * 50)

        else:
            print(f"This experiment is not found on this machine: {exp_name}")


def make_run_overview_csv(run_dir, exp_name):
    overview_csv_file = f"{ANALYSIS_DIR}/{exp_name}_run_overview.csv"

    print(f"Making run overview of {run_dir}, in {overview_csv_file}")

    overview_rows = []
    for run_name in os.listdir(run_dir):
        row = parse_run_name(run_name, exp_name)
        try:
            run_id = get_wandb_run_id(run_name, run_dir=RUN_DIRS[exp_name])
            row['run_id'] = run_id
        except:
            print(f"Could not find run_id for: {run_name}")
            row['run_id'] = None

        # Check efficient epochs if already saved
        if os.path.exists(f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"):
            par_dict = load_pickle(f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p")
            row["efficient_epochs"] = par_dict["efficient_epochs"]
        else:
            row["efficient_epochs"] = None

        # Check efficient epochs if already saved
        if os.path.exists(f"{RES_FILE_DIR}/{exp_name}/{run_name}/full_wandb_pareto_dict.p"):
            full_par_dict = load_pickle(f"{RES_FILE_DIR}/{exp_name}/{run_name}/full_wandb_pareto_dict.p")
            if "pareto epoch" in full_par_dict:
                row["max pareto logged epoch"] = np.max(list(full_par_dict["pareto epoch"].values()))
            else:
                print(f"pareto epoch not in full_par_dict for run: {run_name}")
                row["max pareto logged epoch"] = None
        else:
            row["max pareto logged epoch"] = None

        ckpts = [f for f in os.listdir(f"{run_dir}/{run_name}") if "checkpoint" in f]
        row["n_checkpoints"] = len(ckpts)
        if len(ckpts) == 0:
            print(f"No checkpoints for: {run_name}")
        row["checkpoints"] = ckpts
        overview_rows.append(row)

    run_overview = pd.DataFrame(overview_rows)
    run_overview.to_csv(overview_csv_file)

    return run_overview


def delete_all_pareto_related_files():
    # Delete all pareto related data
    for exp_name in RUN_DIRS:
        if os.path.exists(f"{RES_FILE_DIR}/{exp_name}"):
            for run_name in os.listdir(f"{RES_FILE_DIR}/{exp_name}"):
                if os.path.exists(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
                    for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
                        if "pareto" in f:
                            #                 print(f"{RES_FILE_DIR}/{exp_name}/{run_name}/{f}")
                            p_path = f"{RES_FILE_DIR}/{exp_name}/{run_name}/{f}"
                            os.remove(p_path)


def check_if_running(run_name, exp_name):
    api = wandb.Api()

    run_id = get_wandb_run_id(run_name, run_dir=RUN_DIRS[exp_name])
    wandb_exp = "thesis-test" if exp_name == "Runs-ablation" else "thesis-May"

    run = api.run(f"claartjebarkhof/{wandb_exp}/{run_id}")

    if run.state == "running":
        print(f"STILL RUNNING WARNING: skipping {run_name}, still running")
        return True
    else:
        return False


def save_full_wandb_pareto_dict_to_pickle(run_df, exp_name):
    api = wandb.Api()

    pareto_keys = ["pareto rate", "pareto -D_ks", "pareto -distortion", "pareto iw_ll_x_gen_mean",
                   "pareto iw_ll_mean", "pareto epoch"]

    for row_i, row in run_df.iterrows():
        run_name = row['run_name']
        run_id = row['run_id']

        # print(run_name)

        wandb_exp = "thesis-test" if exp_name == "Runs-ablation" else "thesis-May"

        # Check if already done
        result_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/full_wandb_pareto_dict.p"

        if os.path.isfile(result_pickle):  # uncomment this when everything is done running
            continue

        try:
            run = api.run(f"claartjebarkhof/{wandb_exp}/{run_id}")

            if run.state != "running":
                full_pareto_dict = run.history(keys=pareto_keys).to_dict()

                dump_pickle(full_pareto_dict, result_pickle)

        except Exception as e:
            print(f"** ERROR save_full_wandb_pareto_dict_to_pickle for run {run_name}:", e)


def save_last_pareto_dict_to_pickle(run_overview, run_dir, exp_name):
    for row_i, row in run_overview.iterrows():
        run_name = row["run_name"]
        try:
            # Check if already done
            result_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"

            os.makedirs(f"{RES_FILE_DIR}/{exp_name}/{run_name}", exist_ok=True)

            if os.path.isfile(result_pickle):  # uncomment this when everything is done running
                continue

            # Get the last checkpoint saved
            checkpoint_list = sorted([c for c in os.listdir(f"{run_dir}/{run_name}") if c != "wandb"])
            last_checkpoint = checkpoint_list[-1]

            path = f"{run_dir}/{run_name}/{last_checkpoint}"

            # Load the checkpoint
            c = torch.load(path, map_location='cpu')
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
        elif "AE" in run_name:
            opt = "AE"
            tr = -0.0
        elif "decoder" in run_name.lower():
            opt = "DEC-ONLY"
            tr = -0.0
        else:
            opt = "MDR-0.5"
            tr = 0.5

        # Decoder drop-out
        if "DROP 40" in run_name:
            drop_str = " | DROP 40"
            drop = 0.4
        elif "DROP 60" in run_name:
            drop_str = " | DROP 60"
            drop = 0.6
        elif "DROP 80" in run_name:
            drop_str = " | DROP 80"
            drop = 0.8
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
        if "mdr" in run_name:
            optimisation = "MDR"
        elif "cyc-fb" in run_name:
            optimisation = "CYC-FB"
        elif "fb" in run_name:
            optimisation = "FB"
        elif "VAE" in run_name:
            optimisation = "VAE"
        elif "AE" in run_name:
            optimisation = "AE"
        else:
            optimisation = "CYC"

        if optimisation in ["FB", "MDR"]:
            target_rate = float(run_name.split("-")[6])
        elif optimisation == "CYC-FB":
            target_rate = float(run_name.split("-")[7])
        else:
            target_rate = 0.0

        clean_name = f"PTB | {optimisation} | mem+emb | Target rate: {target_rate}"

        parsed_run_name = {
            "run_name": run_name,
            "mech_string": "mem+emb",
            "dataset_name": "PTB",
            "dataset": "PTB",
            "clean_name": clean_name,
            "target_rate": target_rate,
            "optimisation": optimisation,
            "drop": 0.0,
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
            "drop": 0.0,
            "dataset": "YELP",
            "optimisation": opt_string,
            "mem": True,
            "emb": False,
            "matrix": False
        }

    elif exp_name == "Runs-pretrain":
        if "embeddings" in run_name:
            emb =  True
            mech_string = "mem-emb"
        else:
            emb = False
            mech_string = "mem"

        clean_name = f"PTB | CYC-FB-0.5 | {mech_string} | (YELP pre-trained)"

        parsed_run_name = {
            "clean_name": clean_name,
            "run_name": run_name,
            "drop": 0.0,
            "dataset": "PTB",
            "optimisation": "CYC-FB-0.5",
            "mem": True,
            "emb": emb,
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

        if "decoder" in run_name.lower():
            print(f"Not removing checkpoints from: {run_name}")
            continue

        if check_if_running(run_name, exp_name):
            continue

        least_efficient_epochs = []
        for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
            if "weighted" in f:
                mini_df = pd.read_csv(f"{RES_FILE_DIR}/{exp_name}/{run_name}/{f}", index_col=0)
                # display(mini_df)

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


def get_sum_stats_run(run_name, exp_name="Runs", val_batches=10, iw_samples=200, batch_size=64):
    p = f"{RES_FILE_DIR}/{exp_name}/{run_name}/validation_results_{val_batches}_batches_{iw_samples}_samples_BS_{batch_size}.p"
    return pickle.load(open(p, "rb"))


def calc_runs_missing(run_overview=None):
    if run_overview is None:
        run_overview = read_overview_csv()

    big_exp = {}
    i = 0
    for dataset in ["YELP", "PTB"]:  # "OPTIMUS YELP",
        for drop_out in [0.0, 0.4]:
            drop_str = " | DROP 40" if drop_out == 0.4 else ""
            for optimisation in ["CYC-FB-0.5", "MDR-0.5", "VAE", "AE"]:
                for mech in ["matrix", "matrix+mem", "mem", "mem+emb"]:
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
    # if all values of a stat are the same for all epoch, the resulting scaled score becomes nan
    # in that case, replace with lowest score not to effect anythin (0.0)
    for column in pareto_stats:
        df[column + scale_affix] = df[column + scale_affix].fillna(0.0)
    return df


def calc_weighted_pareto_best_checkpoint(run_name, exp_name, save=True):
    if "dec" in run_name.lower() or " ae " in run_name.lower():
        pareto_stats_weights = {"-distortion": 1}
    else:
        pareto_stats_weights = {
            "iw_ll_mean": 4,
            "iw_ll_x_gen_mean": 3,
            "-D_ks": 1,
            "rate": 4  # for non-collapsed model selection
        }

    scale_affix = "_minmax_norm"

    for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}/"):
        if "weighted_score" in f:
            continue

    pareto_efficient_dict = get_pareto_efficient_dict(run_name, exp_name=exp_name)

    mini_df = pd.DataFrame(pareto_efficient_dict)

    if len(mini_df) == 1:
        for column in pareto_stats_weights.keys():
            mini_df[column + scale_affix] = 1.0
    else:
        mini_df = min_max_scaling(mini_df, list(pareto_stats_weights.keys()), scale_affix)

    mini_df["combined_weighted_score"] = 0.0
    for k, w in pareto_stats_weights.items():
        mini_df["combined_weighted_score"] += w * mini_df[k + scale_affix]

    mini_df = mini_df.sort_values("combined_weighted_score", ascending=False)
    # display(mini_df)

    best_epoch = \
        mini_df[mini_df["combined_weighted_score"] == mini_df["combined_weighted_score"].max()][
            "efficient_epochs"].values[
            0]

    # print("Best epoch", best_epoch)

    path = f"{RES_FILE_DIR}/{exp_name}/{run_name}/weighted_pareto_optimal_point_best[{best_epoch}].csv"
    if save:
        mini_df.to_csv(path)
    else:
        return mini_df


def calc_weighted_pareto_best_checkpoint_all(run_overview, exp_name):
    for row_i, row in run_overview.iterrows():
        run_name = row["run_name"]

        try:
            calc_weighted_pareto_best_checkpoint(run_name, exp_name, save=True)


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
        best_epoch = None

        efficient_epochs = []
        for f in os.listdir(f"{RES_FILE_DIR}/{exp_name}/{run_name}"):
            if "weighted_pareto" in f:

                df = pd.read_csv(f"{RES_FILE_DIR}/{exp_name}/{run_name}/{f}")
                efficient_epochs = df.efficient_epochs.values

        if len(efficient_epochs) == 0:
            print("No efficient epochs found, probably no weighted_pareto file.")
            return None, None

        path = None

        best_epoch = -1
        for e in efficient_epochs:
            # print("efficient epoch", e)
            # Get checkpoint belonging to that epoch
            for f in os.listdir(f"{run_dir}/{run_name}"):
                # print(f)
                if f"epoch-{e:03d}" in f:
                    path = f"{run_dir}/{run_name}/{f}"
                    best_epoch = e
                    break

            if best_epoch != -1:
                break

        print(f"get_best_checkpoint best epoch: {best_epoch}, path: {path}, run_name: {run_name}")

        return path, best_epoch

    else:
        print("no weighted pareto scores for this run yet", run_name)
        return None, None


def get_pareto_efficient_dict(run_name, exp_name="Runs"):
    pareto_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"

    if not os.path.isfile(pareto_pickle):
        print(pareto_pickle)
        print("get_pareto_efficient_dict: Could not find pareto pickle, try running update().")
        return None

    pareto_dict = pickle.load(open(pareto_pickle, "rb"))
    pareto_eff_dict = {"efficient_epochs": pareto_dict["efficient_epochs"]}

    # print("run_name", run_name)
    # print("pareto_eff_dict", pareto_eff_dict)

    if "decoder" in run_name.lower() or " ae " in run_name.lower():
        keys = ["-distortion"]
    else:
        keys = ["iw_ll_mean", "iw_ll_x_gen_mean", "-D_ks", "rate", "-distortion"]

    for i, name in enumerate(keys):
        #print(name)
        pareto_eff_dict[name] = []
        for e in pareto_dict["efficient_epochs"]:
            pareto_eff_dict[name].append(pareto_dict[name][e])

    return pareto_eff_dict


def plot_pareto_stats(run_name, clean_name=None, best_epoch=None, exp_name="Runs"):
    last_pareto_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/last_pareto_dict.p"

    full_pareto_pickle = f"{RES_FILE_DIR}/{exp_name}/{run_name}/full_wandb_pareto_dict.p"

    if not os.path.isfile(full_pareto_pickle) or not os.path.isfile(last_pareto_pickle):
        print("plot_pareto_stats: Could not find full or last pareto pickle, try running update().")
        return None

    if best_epoch is None:
        _, best_epoch = get_best_checkpoint(run_name=run_name, exp_name=exp_name)

    full_pareto_dict = pickle.load(open(full_pareto_pickle, "rb"))
    last_pareto_dict = pickle.load(open(last_pareto_pickle, "rb"))

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 4.5))

    if clean_name is None:
        clean_name = run_name

    fig.suptitle(clean_name, y=1.05, size=12)

    for i, name in enumerate(["pareto iw_ll_mean", "pareto iw_ll_x_gen_mean", "pareto -D_ks"]):
        axs[0, i].plot(list(full_pareto_dict[name].values()))
        axs[0, i].set_title(name)

        for e in last_pareto_dict["efficient_epochs"]:
            key = name.split("pareto ")[-1]
            axs[0, i].scatter(e, last_pareto_dict[key][e], color='r')

        if best_epoch is not None:
            axs[0, i].scatter(best_epoch, full_pareto_dict[name][best_epoch], color='g')


        else:
            print(f"XXXX no best epoch for : {clean_name}")

    for i, name in enumerate(["pareto rate", "pareto -distortion"]):
        axs[1, i].plot(list(full_pareto_dict[name].values()))
        axs[1, i].set_title(name)

        for e in last_pareto_dict["efficient_epochs"]:
            key = name.split("pareto ")[-1]
            axs[1, i].scatter(e, last_pareto_dict[key][e], color='r')

        if best_epoch is not None:
            axs[1, i].scatter(best_epoch, full_pareto_dict[name][best_epoch], color='g')
            if "rate" in name and full_pareto_dict[name][best_epoch] < 1.0 and not "VAE" in run_name:
                print(f"YYYY collapsed model warning for: {clean_name}")

    plt.delaxes(ax=axs[1, 2])
    plt.tight_layout()
    plt.show()


update()
