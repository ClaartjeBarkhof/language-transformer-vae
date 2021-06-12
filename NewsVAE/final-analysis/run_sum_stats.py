import sys; sys.path.append("/home/cbarkhof/code-thesis/NewsVAE")
import sys; sys.path.append("/home/cbarkhof/code-thesis/NewsVAE/final-analysis")

from utils_train import load_from_checkpoint, transfer_batch_to_device
from utils_analysis import *
from dataset_wrappper import NewsData
import torch
from loss_and_optimisation import approximate_log_q_z

BATCH_SIZE = 64
IW_N_SAMPLES = 300
VAL_BATCHES = 20
DEVICE = "cuda:0"

for exp_name, run_dir in RUN_DIRS.items():

    print("EXP:", exp_name)

    runs_df = read_overview_csv(exp_name=exp_name)

    for row_idx, row in runs_df.iterrows():
        run_name = row["run_name"]
        print("*" * 50)
        print(run_name)
        print("*" * 50)

        try:
            # --------------------------------------------------
            # Save the stats to
            result_file = f"{RES_FILE_DIR}/{exp_name}/{run_name}/validation_results_{VAL_BATCHES}_batches_{IW_N_SAMPLES}_samples_BS_{BATCH_SIZE}.p"
            if os.path.isfile(result_file):
                print("Did this one already, continuing.")
                continue

            # --------------------------------------------------
            # Get data that this model was trained on
            if "optimus" in row["dataset"].lower():
                dataset_name = "optimus_yelp"
            elif "yelp" in row["dataset"].lower():
                dataset_name = "yelp"
            else:
                dataset_name = "ptb_text_only"

            # Load relevant data
            data = NewsData(dataset_name=dataset_name, tokenizer_name="roberta", batch_size=BATCH_SIZE,
                            num_workers=3, pin_memory=True, max_seq_len=64, device=DEVICE)
            val_loader = data.val_dataloader(shuffle=False, batch_size=BATCH_SIZE)

            # Determine number of batches to evaluate
            if VAL_BATCHES > len(val_loader) or VAL_BATCHES < 0:
                MAX_BATCHES = len(val_loader)
            else:
                MAX_BATCHES = VAL_BATCHES

            # --------------------------------------------------
            # Get best checkpoint and loss term manager
            path, best_epoch = get_best_checkpoint(run_name, exp_name=exp_name)
            loss_term_manager = load_from_checkpoint(path, world_master=True, ddp=False, device_name="cuda:0",
                                                     evaluation=True, return_loss_term_manager=True)

            res = {
                "lens_x_gen": [],  # batch
                "lens": [],

                "iw_ll_x_gen_p_w": [],  # batch
                "iw_ll_x_gen": [],
                "iw_ll_p_w": [],
                "iw_ll": [],

                "reconstruction_loss": [],  # float
                "exact_match": [],
                "cross_entropy_per_word": [],

                "log_q_z_x": [],  # batch x samples
                "log_p_z": [],
                "log_p_x_z": [],

                "latent_z": [],  # batch x latent_dim
                "mu": [],
                "logvar": [],

                "kl_analytical": [],  # float
                "fb_kl_analytical": [],
                "elbo": [],
                "marginal KL": [],
                "dim_KL": [],
                "TC": [],
                "MI": [],
                "mmd_loss": [],

                "std_x_std": [],  # float
                "mean_x_std": [],
                "std_x_mu": [],
                "mean_x_mu": [],
                "std_z_std": [],
                "std_z_mu": [],
            }

            # --------------------------------------------------
            # Make predictions
            for batch_i, batch in enumerate(val_loader):
                print(f"Batch {batch_i + 1:3d}/{MAX_BATCHES:3d}", end="\r")

                batch = transfer_batch_to_device(batch)

                with torch.no_grad():
                    out = loss_term_manager(batch["input_ids"], batch["attention_mask"],
                                            return_exact_match=True, decoder_only=False, eval_iw_ll_x_gen=True,
                                            return_posterior_stats=True, device_name=DEVICE,
                                            iw_ll_n_samples=IW_N_SAMPLES,
                                            return_attention_to_latent=False, train=False, max_seq_len_x_gen=64)

                for k, v in out.items():
                    if k in res:
                        if torch.is_tensor(v):
                            res[k].append(v.cpu())
                        else:
                            res[k].append(v)

                if batch_i + 1 == MAX_BATCHES:
                    break

            # --------------------------------------------------
            # Gather predictions
            for k, v in res.items():
                if torch.is_tensor(v[0]):
                    if v[0].dim() > 0:
                        res[k] = torch.cat(v).cpu()
                    else:
                        res[k] = torch.stack(v).cpu()

            # --------------------------------------------------
            # Add log q z on whole validation set (only one latent sample z per data point x is needed)
            res["log_q_z"], _ = approximate_log_q_z(res["mu"], res["logvar"], res["latent_z"][:, 0, :], method="all",
                                                    prod_marginals=False, reduce_mean=False)
            res["log_q_z_mean"] = res["log_q_z"].mean()

            # --------------------------------------------------
            # Dump the result in a pickle
            pickle.dump(res, open(result_file, "wb"))

        except Exception as e:
            print("** ERROR for :", run_name, e)