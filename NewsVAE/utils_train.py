import platform
import socket
import collections
import datetime
import os
import wandb
import arguments
import numpy as np
from scipy import stats
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scipy import stats
from utils_external import tie_weights
import modules.vae as vae

from dataset_wrappper import NewsData

# ----------------------------------------------------------------------------------------------------
# INITIALISATION STUFF
# ----------------------------------------------------------------------------------------------------

def set_device(device_rank):
    """
    Set the device in case of cuda and give the device a name for .to(...) operations.

    Args:
        device_rank: Union[str, int]
            GPU rank if it is an integer, else "cpu" string.
    Returns:
        device_name: str:
            A device name that can be used with .to(device)
    """
    # GPU
    if type(device_rank) == int:
        print(f"-> Setting device {device_rank}")
        torch.cuda.set_device(device_rank)
        cudnn.benchmark = True  # optimise backend algo
        device_name = f"cuda:{device_rank}"

    # CPU
    else:
        device_name = "cpu"
    return device_name


def set_ddp_environment_vars(port_nr=1235):
    """
    Set the environment variables for the DDP environment.

    Args:
        port_nr: int: the port to use (default: 1234)
    """
    os.environ['MASTER_ADDR'] = get_ip()
    os.environ['MASTER_PORT'] = str(port_nr)
    print("Setting MASTER_ADDR: {}, MASTER PORT: {}...".format(get_ip(), port_nr))


def get_run_name(run_name_prefix=""):
    """
    Make a unique run name, with the current type and possibly some named prefix.

    Args:
        run_name_prefix: str
    """

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    date, time = datetime_stamp.split("--")[0], datetime_stamp.split("--")[1]
    run_name = "{}-{}-run-{}".format(date, run_name_prefix, time)
    print("Run is called: {}...".format(run_name))
    return run_name


def prepare_folders(run_name, code_path):
    """
    Make folders to save stuff in.

    Args:
        run_name: str
        code_path: str:
            The path to the NewsVAE code directory (dependent on the machine)
    """

    os.makedirs('{}/Runs/{}'.format(code_path, run_name), exist_ok=True)
    os.makedirs('{}/Runs/{}/wandb'.format(code_path, run_name), exist_ok=True)


def determine_global_max_steps(max_global_train_steps, batch_size, world_size, accumulate_n_batches_grad):
    effective_batch_size = world_size * accumulate_n_batches_grad * batch_size
    max_global_grad_steps_rank = int(max_global_train_steps / accumulate_n_batches_grad)

    print(f"Max global steps on this rank: {max_global_train_steps}")
    print(f"This amounts to {max_global_grad_steps_rank} max global gradient steps on this rank.")
    print(f"The effective batch size is {effective_batch_size} (world_size x grad_acc_steps x batch_size = "
          f"{world_size} x {accumulate_n_batches_grad} x {batch_size} = {effective_batch_size}).")

    return max_global_train_steps, max_global_grad_steps_rank


def get_world_specs(n_gpus, n_nodes, device_name):
    """
    Determine the world size and whether the current device is world master.

    Args:
        n_gpus: int
        n_nodes: int
        device_name: str

    Returns:
        world_size: int
            How many devices are active
        world_master: bool
            Whether the current device is world master (GPU 0 or CPU)
    """

    world_size = int(n_gpus * n_nodes) if "cuda" in device_name else 1
    world_master = True if device_name in ["cuda:0", "cpu"] else False

    return world_master, world_size


def determine_max_epoch_steps_per_rank(max_train_steps_epoch_per_rank, max_valid_steps_epoch_per_rank, datasets_dict,
                                       batch_size, world_size=1, world_master=True):
    """
    Determine epoch length by applying logic to the configuration and
    the size of the dataset.

    First determine how many batches are present in the dataset (steps).
    Then see how many there are per rank (/ world size).
    Then determine whether that is more than set in the settings. If so, set to maximum.
    If set to -1, set to maximum as well.

    Args:

    """
    assert "train" in datasets_dict and "validation" in datasets_dict, \
        "-> Expects a dataset with both train and validation keys present."

    # How many batches are there in the dataset per rank
    n_train_batches_per_rank = int(len(datasets_dict['train']) / (world_size * batch_size))
    n_valid_batches_per_rank = int(len(datasets_dict['validation']) / (world_size * batch_size))

    if world_master:
        print(f"Length of the train set: {len(datasets_dict['train'])}, which amounts to max. "
              f"{n_train_batches_per_rank} batches of size {batch_size} per rank.")
        print(f"Length of the validation set: {len(datasets_dict['validation'])}, which amounts to max. "
              f"{n_valid_batches_per_rank} batches of size {batch_size} per rank.")
        print(f"max_train_steps_epoch_per_rank in config set to: "
              f"{max_train_steps_epoch_per_rank}")
        print(f"max_valid_steps_epoch_per_rank in config set to: "
              f"{max_valid_steps_epoch_per_rank}")

    # Train
    if max_train_steps_epoch_per_rank == -1:
        max_train_steps_epoch_per_rank = n_train_batches_per_rank
        if world_master: print(f"Setting max_train_steps_epoch to {n_train_batches_per_rank} "
                               f"given a world size of {world_size}")
    else:
        if max_train_steps_epoch_per_rank > n_train_batches_per_rank:
            if world_master:
                print("Warning: multiple passes over the TRAIN data are "
                      "contained in the number of steps per epoch set. "
                      "Setting max steps to epoch length.")
            max_train_steps_epoch_per_rank = n_train_batches_per_rank

    # Validation
    if max_valid_steps_epoch_per_rank == -1:
        if world_master: print(f"Setting max_train_steps_epoch to {n_train_batches_per_rank} "
                               f"given a world size of {world_size}")
        max_valid_steps_epoch_per_rank = n_valid_batches_per_rank
    else:
        if max_valid_steps_epoch_per_rank > n_valid_batches_per_rank:
            print("Warning: multiple passes over the VALIDATION data are "
                  "contained in the number of steps per epoch set. "
                  "Setting max steps to epoch length.")
            max_valid_steps_epoch_per_rank = n_valid_batches_per_rank

    return max_train_steps_epoch_per_rank, max_valid_steps_epoch_per_rank

# ----------------------------------------------------------------------------------------------------
# DATA LOADERS
# ----------------------------------------------------------------------------------------------------

def get_dataloader(phases, ddp=False, batch_size=12, num_workers=8, max_seq_len=64, world_size=4,
                   dataset_name="cnn_dailymail", tokenizer_name="roberta",
                   device_name="cuda:0", world_master=True, gpu_rank=0):
    """
    Get data loaders for distributed sampling for different phases (eg. train and validation).

    Args:
        phases: list:
            List of phases to make dataloaders for (train, validation, test)
        ddp: bool:
            Whether or not to use Distributed Data Parallel
        batch_size: int
        num_workers: int
        max_seq_len: int
        world_size: int
            Number of nodes x number of GPUs
        dataset_name: str:
            The name of the dataset ("cnn_dailymail")
        tokenizer_name: str:
            The name of the tokenizer ("roberta")
        device_name: str:
            Which device is active.
        world_master: bool:
            Whether or not to print.
        gpu_rank: int:
            If cuda, what the rank of the GPU is.

    Returns:
        loaders: dict[str, torch.utils.data.DataLoader]
            A dict with dataloaders for every named phase.
        data:
            NewsData object with the data # TODO: change this
        samplers: dict[str. torch.utils.data.distributed.DistributedSampler]
            A dict with samplers for the GPU (if ddp).
    """

    if world_master: print("Get dataloaders...")

    pin_memory = True if "cuda" in device_name else False

    loaders = {}
    samplers = {}

    # Get data
    data = NewsData(dataset_name, tokenizer_name,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, max_seq_len=max_seq_len,
                    device=device_name)

    for phase in phases:
        # DDP
        if ddp:
            sampler = DistributedSampler(data.datasets[phase], num_replicas=world_size,
                                         shuffle=True, rank=gpu_rank)
            samplers[phase] = sampler

            # With distributed sampling, shuffle must be false, the sampler takes care of that
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        sampler=sampler, pin_memory=True, shuffle=False,
                                        num_workers=num_workers, collate_fn=data.collate_fn)
        # Single GPU and CPU
        else:
            # Without distributed sampling
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        shuffle=True, pin_memory=pin_memory,
                                        num_workers=num_workers, collate_fn=data.collate_fn)

    if world_master: print("Done getting dataloaders...")

    return loaders, data, samplers


# ----------------------------------------------------------------------------------------------------
# LOAD + SAVE MODEL & CHECKPOINTING
# ----------------------------------------------------------------------------------------------------

def fix_query_key_value_layer_name(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "decoder" in k:
            if "self.query" in k:

                new_name = k.replace("self.query", "self.query_module.layer")
                #print(f"replacing {k} with {new_name}")
                new_state_dict[new_name] = v
            elif "self.key" in k:
                new_name = k.replace("self.key", "self.key_module.layer")
                new_state_dict[new_name] = v
            elif "self.value" in k:
                new_name = k.replace("self.value", "self.value_module.layer")
                new_state_dict[new_name] = v

        new_state_dict[k] = v

    return new_state_dict


def load_from_checkpoint(path, world_master=True, ddp=False, device_name="cuda:0", latent_size=32,
                         do_tie_embeddings=True, do_tie_weights=True, add_latent_via_memory=True,
                         add_latent_via_gating=False, add_latent_via_cross_attention=False,
                         add_latent_via_embeddings=False, do_tie_embedding_spaces=True, dataset_size=3370,
                         add_decoder_output_embedding_bias=False, objective="evaluation", evaluation=True,
                         return_loss_term_manager=False):
    # DETERMINE / CHECK PATH
    assert os.path.isfile(path), f"-> checkpoint file path ({path}) must exist for it to be loaded!"
    if world_master: print("Loading model from checkpoint: {}".format(path))

    # LOAD CHECKPOINT
    checkpoint = torch.load(path, map_location='cpu')

    # Get standard config
    config = arguments.preprare_parser(jupyter=True, print_settings=False)

    if "config" in checkpoint:
        print("found a config in the checkpoint!")
        for k, v in vars(checkpoint["config"]).items():
            if k in vars(config):
                setattr(config, k, v)
    else:
        overview_file = "/home/cbarkhof/code-thesis/NewsVAE/runs_overview/overview_main.csv"
        run_name = path.split("/")[-2]

        print("Looking for run in overview:", run_name)

        if os.path.isfile(overview_file):
            run_df = pd.read_csv(overview_file, delimiter=";")
            mechanism = run_df[run_df["Run name"] == run_name]['Latent mechanism'].values[0]

            print("Found mechanism:", mechanism)

            if mechanism == "Memory" or mechanism == "Memory + Embeddings":
                config.add_latent_via_memory = True
            else:
                config.add_latent_via_memory = False

            if mechanism == "Memory + Embeddings" or mechanism == "Embeddings":
                config.add_latent_via_embeddings = True
            else:
                config.add_latent_via_embeddings = False

            if mechanism == "Gating":
                config.add_latent_via_gating = True
            else:
                config.add_latent_via_gating = False

            if mechanism == "Cross-attention":
                config.add_latent_via_cross_attention = True
            else:
                config.add_latent_via_cross_attention = False
        else:

            config.do_tie_weights = do_tie_weights
            config.objective = objective
            config.latent_size = latent_size
            config.add_latent_via_memory = add_latent_via_memory
            config.add_latent_via_gating = add_latent_via_gating
            config.add_latent_via_cross_attention = add_latent_via_cross_attention
            config.add_latent_via_embeddings = add_latent_via_embeddings
            config.do_tie_embedding_spaces = do_tie_embedding_spaces
            config.add_decoder_output_embedding_bias = add_decoder_output_embedding_bias

    if return_loss_term_manager:
        loss_term_manager = vae.get_loss_term_manager_with_model(config, world_master=True,
                                            dataset_size=dataset_size, device_name=device_name)
        vae_model = loss_term_manager.vae_model
    else:
        vae_model = vae.get_model_on_device(config, dataset_size=dataset_size, device_name=device_name, world_master=True)

    # Bring to CPU, as state_dict loading needs to happen in CPU (strange memory errors occur otherwise)
    vae_model = vae_model.cpu()

    # DDP vs no DDP
    parameter_state_dict = checkpoint["VAE_model_state_dict"]


    # if config.add_latent_w_matrix_influence is True:
    #     print("TEST")
    parameter_state_dict = fix_query_key_value_layer_name(parameter_state_dict)

    # MODEL
    if "module." in list(checkpoint["VAE_model_state_dict"].keys())[0] and not ddp:
        print("Removing module string from state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(parameter_state_dict, remove=True)

    elif "module." not in list(checkpoint["VAE_model_state_dict"].keys())[0] and ddp:
        print("Adding module string to state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(parameter_state_dict, remove=False)

    # in place procedure
    vae_model.load_state_dict(parameter_state_dict)
    vae_model = vae_model.to(device_name)

    # GLOBAL STEP, EPOCH, BEST VALIDATION LOSS
    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]
    best_valid_loss = checkpoint["best_valid_loss"]

    print("Checkpoint global_step: {}, epoch: {}, best_valid_loss: {}".format(global_step,
                                                                              epoch,
                                                                              best_valid_loss))

    # Do this again after loading checkpoint, because I am not sure if that is saved correctly
    # Weight tying / sharing between encoder and decoder RoBERTa part
    if do_tie_weights and config.decoder_only is False:
        print("Tying encoder decoder RoBERTa checkpoint weights!")
        base_model_prefix = vae_model.decoder.model.base_model_prefix
        tie_weights(vae_model.encoder.model, vae_model.decoder.model._modules[base_model_prefix], base_model_prefix)

    # Make all embedding spaces the same (encoder input, decoder input, decoder output)
    if do_tie_embeddings and config.decoder_only is False:
        print("Tying embedding spaces!")
        vae_model.tie_all_embeddings()

    if evaluation is True:
        print("Setting to eval mode.")
        vae_model.eval()

    if return_loss_term_manager:
        loss_term_manager.vae_model = vae_model
        return loss_term_manager
    else:
        return vae_model

# Code for the fn below is taken from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """

    NB: I changed the function from costs to gains, changing the condition to a greater than.

    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost (changed to greater than, for gain)
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True  # And keep self
    return is_efficient


def determine_checkpoint(val_epoch_stats, epoch_pareto_effiency_dict, epoch):
    """

    Args:
        val_epoch_stats: a dict with metrics per epoch over batches, structured as follows:
            val_epoch_stats[epoch_i][metric_i] = list of metric values for batches

        epoch_pareto_effiency_dict: a summary per epoch of a selection of metric to keep track of the
            pareto frontier measured along those dimensions, the dimensions are defined as keys in the dict:
                rate, -distortion, iw_ll_mean, iw_ll_x_gen_mean, -D_ks
                -> the values are the mean of the metric at the epoch corresponding to the index in the list
                -> some are negated to make the direction higher is better for all metrics
    Returns:
        epoch_pareto_effiency_dict: with the stats of the current epoch added



    """

    iw_ll, iw_ll_x_gen = val_epoch_stats["iw_ll_p_w"], val_epoch_stats["iw_ll_x_gen_p_w"]
    ks_statistic, pval = stats.ks_2samp(iw_ll, iw_ll_x_gen)

    iw_ll_mean = np.mean(iw_ll)
    iw_ll_x_gen_mean = np.mean(iw_ll_x_gen)
    rate_mean = np.mean(val_epoch_stats["kl_analytical"])
    min_distortion_mean = - np.mean(val_epoch_stats["reconstruction_loss"])
    min_d_ks = - ks_statistic

    epoch_pareto_effiency_dict["rate"].append(rate_mean)
    epoch_pareto_effiency_dict["-distortion"].append(min_distortion_mean)
    epoch_pareto_effiency_dict["-D_ks"].append(min_d_ks)
    epoch_pareto_effiency_dict["iw_ll_mean"].append(iw_ll_mean)
    epoch_pareto_effiency_dict["iw_ll_x_gen_mean"].append(iw_ll_x_gen_mean)

    # Change the dict to a set of array of multi dimensional points
    multi_dim_points = np.asarray([[
        epoch_pareto_effiency_dict["rate"][i],
        epoch_pareto_effiency_dict["-distortion"][i],
        epoch_pareto_effiency_dict["-D_ks"][i],
        epoch_pareto_effiency_dict["iw_ll_mean"][i],
        epoch_pareto_effiency_dict["iw_ll_x_gen_mean"][i]] for i in range(epoch+1)])

    efficient_epochs = is_pareto_efficient_simple(multi_dim_points)
    efficient_epochs = np.where(efficient_epochs)[0].tolist() # convert boolean array to list with epoch indices

    return epoch_pareto_effiency_dict, efficient_epochs

    # print("iw_ll mean", iw_ll_mean)
    # print("iw_ll x_gen mean", iw_ll_x_gen_mean)
    # print("ks statistic, pvalue:", ks_statistic, pval)
    # quit()

    # if 'elbo' in val_epoch_stats:
    #     mean_valid_loss = - np.mean(val_epoch_stats['elbo'])  # <- select - ELBO
    # else:
    #     mean_valid_loss = np.mean(val_epoch_stats['total_loss'])  # <- select on total loss
    #
    # mean_valid_rate = np.mean(val_epoch_stats["kl_analytical"])
    # mean_valid_rec = np.mean(val_epoch_stats["reconstruction_loss"])
    #
    # # Better ELBO, lower rec and higher rate
    # if mean_valid_rec < best_valid_rec and mean_valid_rate > best_valid_rate and mean_valid_loss < best_valid_loss:
    #     print(f"Found a better model in terms of ELBO and in terms of rate / distortion (non-collapsed model) on this device: "
    #           f"elbo: {mean_valid_loss:.2f} rate: {mean_valid_rate:.2f}, {mean_valid_rec:.2f}. Saving model!")
    #     checkpoint_type = "best-elbo-rate-distortion"
    #     best_valid_rate = mean_valid_rate
    #     best_valid_rec = mean_valid_rec
    #     best_valid_loss = mean_valid_loss
    #
    # # Lower rec, higher rate
    # elif mean_valid_rec < best_valid_rec and mean_valid_rate > best_valid_rate:
    #     print(f"Found a better model in terms of rate / distortion (non-collapsed model) on this device: "
    #           f"rate: {mean_valid_rate:.2f}, {mean_valid_rec:.2f}. Saving model!")
    #     checkpoint_type = "best-rate-distortion"
    #     best_valid_rate = mean_valid_rate
    #     best_valid_rec = mean_valid_rec
    #
    # # Better ELBO
    # elif mean_valid_loss < best_valid_loss:
    #     print(f"Found better model in terms of ELBO on this device: {mean_valid_loss:.2f}. Saving model!")
    #     checkpoint_type = "best-elbo"
    #     best_valid_loss = mean_valid_loss
    #
    # else:
    #     checkpoint_type = "no-checkpoint"
    #
    # return checkpoint_type, best_valid_loss, best_valid_rate, best_valid_rec


def save_checkpoint_model(vae_model, run_name, code_dir_path, global_step,
                          current_epoch, config, efficient_epochs, epoch_pareto_effiency_dict):
    """
    Save checkpoint for later use.
    """

    # Check which checkpoints are saved, but no longer Pareto efficient
    rmv_ckpts = []
    epochs = []
    for ckpt in os.listdir('{}/Runs/{}'.format(code_dir_path, run_name)):
        # format: name = "checkpoint-epoch-02-step-1000-iw-ll-100.pth"
        e = int(ckpt.split('-')[2])
        epochs.append(e)
        rmv_ckpts.append(e)

    # Remove the checkpoints that are no longer Pareto efficient
    for c in rmv_ckpts:
        print(f"Removing {c}, no longer Pareto efficient.")
        os.remove(f"{code_dir_path}/Runs/{run_name}/{c}")

    # If new efficient checkpoint is found, save it as such
    if current_epoch in efficient_epochs:
        min_iw_ll = abs(int(epoch_pareto_effiency_dict["iw_ll_mean"][-1]))
        ckpt_name = f"checkpoint-epoch-{current_epoch:03d}-step-{global_step}-iw-ll_{min_iw_ll:3d}.pth"
        ckpt_path = '{}/Runs/{}/'.format(code_dir_path, run_name, ckpt_name)
        print(f"Saving checkpoint at {ckpt_path}")

        # TODO: save scaler, scheduler, optimisers for continue training
        checkpoint = {
            'VAE_model_state_dict': vae_model.state_dict(),
            "epoch_pareto_effiency_dict": {"efficient_epochs": efficient_epochs, **epoch_pareto_effiency_dict},
            "config": config,
            # 'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            # 'scheduler_state_dict': scheduler.state_dict(),
            # 'scaler_state_dict': scaler.state_dict(),
            'epoch': current_epoch,
        }

        torch.save(checkpoint, ckpt_path)


def add_remove_module_from_state_dict(state_dict, remove=True):
    """
    Adds or removes the 'module' string from keys in state dict that
    appears after saving in ddp mode.

    Args:
        state_dict: Dict[str, Tensor]
            parameter state dict
        remove: bool
            whether to remove (True) or add (False) the module string
    Returns:
        new_state_dict: Dict[str, Tensor]
            parameter state dict with module strings removed from
            or added from keys
    """

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if remove:
            name = k[7:]  # remove `module.`
        else:
            name = "module." + k
        new_state_dict[name] = v

    return new_state_dict


# ----------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------

def add_matrix_influence_weight_to_loss(loss_term_manager, global_step, global_grad_step, ddp=False):
    gate_weights = {
        "global_step": global_step,
        "global_grad_step": global_grad_step
    }

    if ddp is True:
        model = loss_term_manager.module.vae_model
    else:
        model = loss_term_manager.vae_model

    for l_i, l in enumerate(model.decoder.model.roberta.encoder.layer):
        gate_weights[f"avg_query_{l_i}"] = l.attention.self.query_module.current_gate_weight_mean
        gate_weights[f"avg_key_{l_i}"] = l.attention.self.key_module.current_gate_weight_mean
        gate_weights[f"avg_value_{l_i}"] = l.attention.self.value_module.current_gate_weight_mean

        gate_weights[f"std_query_{l_i}"] = l.attention.self.query_module.current_gate_weight_std
        gate_weights[f"std_key_{l_i}"] = l.attention.self.key_module.current_gate_weight_std
        gate_weights[f"std_value_{l_i}"] = l.attention.self.value_module.current_gate_weight_std

        gate_weights[f"query_{l_i}_original_matrix_norm"] = l.attention.self.query_module.orginal_matrix_norm
        gate_weights[f"key_{l_i}_original_matrix_norm"] = l.attention.self.key_module.orginal_matrix_norm
        gate_weights[f"value_{l_i}_original_matrix_norm"] = l.attention.self.value_module.orginal_matrix_norm

        gate_weights[f"query_{l_i}_avg_predicted_matrix_norm"] = l.attention.self.query_module.avg_predicted_matrix_norm
        gate_weights[f"key_{l_i}_avg_predicted_matrix_norm"] = l.attention.self.key_module.avg_predicted_matrix_norm
        gate_weights[f"value_{l_i}_avg_predicted_matrix_norm"] = l.attention.self.value_module.avg_predicted_matrix_norm

    N_layers = len(model.decoder.model.roberta.encoder.layer)

    gate_weights["average_avg_query_gate_score"] = np.mean([gate_weights[f"avg_query_{i}"] for i in range(N_layers)])
    gate_weights["average_avg_key_gate_score"] = np.mean([gate_weights[f"avg_key_{i}"] for i in range(N_layers)])
    gate_weights["average_avg_value_gate_score"] = np.mean([gate_weights[f"avg_query_{i}"] for i in range(N_layers)])

    gate_weights["average_std_query_gate_score"] = np.mean([gate_weights[f"std_query_{i}"] for i in range(N_layers)])
    gate_weights["average_std_key_gate_score"] = np.mean([gate_weights[f"std_key_{i}"] for i in range(N_layers)])
    gate_weights["average_std_value_gate_score"] = np.mean([gate_weights[f"std_query_{i}"] for i in range(N_layers)])

    gate_weights["average_query_avg_predicted_matrix_norm"] = np.mean(
        [gate_weights[f"query_{i}_avg_predicted_matrix_norm"] for i in range(N_layers)])
    gate_weights["average_key_avg_predicted_matrix_norm"] = np.mean(
        [gate_weights[f"key_{i}_avg_predicted_matrix_norm"] for i in range(N_layers)])
    gate_weights["average_value_avg_predicted_matrix_norm"] = np.mean(
        [gate_weights[f"value_{i}_avg_predicted_matrix_norm"] for i in range(N_layers)])
    gate_weights["average_query_original_matrix_norm"] = np.mean(
        [gate_weights[f"query_{i}_original_matrix_norm"] for i in range(N_layers)])
    gate_weights["average_value_original_matrix_norm"] = np.mean(
        [gate_weights[f"value_{i}_original_matrix_norm"] for i in range(N_layers)])
    gate_weights["average_key_original_matrix_norm"] = np.mean(
        [gate_weights[f"key_{i}_original_matrix_norm"] for i in range(N_layers)])

    wandb.log(gate_weights)

def stats_over_sequence(list_of_stat_batches, list_of_mask_batches, with_relative_positions=True, N_bins=-1):
    assert list_of_stat_batches[0].shape == list_of_mask_batches[0].shape, "stats blocks and mask blocks need to be of equal size"

    acc_stats = cat_pad_uneven(list_of_stat_batches, pad_value=0)
    masks = cat_pad_uneven(list_of_mask_batches, pad_value=0)
    seq_lens = masks.sum(dim=1)

    n_samples, max_len = acc_stats.shape
    if N_bins == -1:
        N_bins = max_len
    positions = torch.arange(1, max_len + 1).unsqueeze(0).repeat(n_samples, 1)
    relative_positions = positions / seq_lens.unsqueeze(1)

    stats_masked = torch.masked_select(acc_stats, masks == 1.0).tolist()
    absolute_positions_masked = torch.masked_select(positions, masks == 1.0)
    relative_positions_masked = torch.masked_select(relative_positions, masks == 1.0)

    if with_relative_positions is True:
        positions = relative_positions_masked.tolist()
        bin_means, bin_edges, bin_ids = stats.binned_statistic(positions,
                                                               stats_masked,
                                                               statistic='mean', bins=N_bins)
    else:
        positions = absolute_positions_masked.tolist()
        bin_means, bin_edges, bin_ids = stats.binned_statistic(positions,
                                                               stats_masked,
                                                               statistic='mean', bins=N_bins)


    return bin_means, bin_edges, bin_ids, stats_masked, positions


def init_logging(vae_model, run_name, code_dir_path, wandb_project, config):
    """
    Initialise W&B logging.

    Args:
        vae_model: NewsVAE: the model object
        run_name: str: name of the run
        code_dir_path: str:
            Path to NewsVAE code dir
        wandb_project: str
            Name of the W&B project.
        config: Namespace: configuration to save
    """

    print("Initialising W&B logging...")
    wandb.init(name=run_name, project=wandb_project,
               dir='{}/Runs/{}/wandb'.format(code_dir_path, run_name), config=config)
    wandb.watch(vae_model)


def log_stats_epoch(stats, epoch, global_step, global_grad_step, atts_to_latent, masks):
    """
    Log mean stats of epoch to W&B.

    Args:
        stats: defaultDict
            Nested dict containing all the statistics that are kept track of.
        epoch: int
        global_step: int
        global_grad_step: int
    """
    logs = {}

    if len(masks) > 0:
        bin_means, bin_edges, bin_ids, stats_masked, _ = stats_over_sequence(atts_to_latent, masks,
                                                                             with_relative_positions=True, N_bins=-1)
        avg_first_three_bins = np.mean(bin_means[:3])
        avg_last_three_bins = np.mean(bin_means[-3:])
        data = [[x, y] for (x, y) in zip(bin_edges, bin_means)]
        table = wandb.Table(data=data, columns=["relative_positions", "avg_att_to_latent"])
        wandb.log({f"attention_to_latent_epoch_{epoch}": wandb.plot.line(table,
                  "relative_positions", "avg_att_to_latent", title="Avg attention to latent with relative positions")})

        logs["Epoch mean (validation) avg_attention_to_latent"] = np.mean(stats_masked)
        logs["Epoch mean (validation) FIRST_three_bins avg_attention_to_latent"] = avg_first_three_bins
        logs["Epoch mean (validation) LAST_three_bins avg_attention_to_latent"] = avg_last_three_bins
        logs["Epoch mean (validation) DIFF FIRST LAST three_bins avg_attention_to_latent"] = avg_first_three_bins - avg_last_three_bins
        logs["Epoch std (validation) std_attention_to_latent"] = np.std(stats_masked)

    for phase, phase_stats in stats[epoch].items():
        print("phase", phase)
        for stat_name, stat in phase_stats.items():
            #print(stat_name)
            if stat_name not in ["iw_ll_x_gen", "iw_ll", "iw_ll_p_w", "iw_ll_x_gen_p_w", "lens", "lens_x_gen"]:
                log_name = "Epoch mean ({}) {}".format(phase, stat_name)
                logs[log_name] = np.mean(stat)

            else:
                # for iw_ll, and iw_ll_x_gen make a histogram (already list type)
                print("log epoch stats", stat_name, len(stat), stat)
                logs[stat_name] = wandb.Histogram(stat)

    logs['epoch'] = epoch
    logs['global step'] = global_step
    logs['global grad step'] = global_grad_step
    logs["custom_step"] = global_grad_step
    wandb.log(logs)


def log_losses_step(losses, phase, epoch, log_every_n_steps, global_step, global_grad_step):
    """
    Log losses of step to W&B.

    Args:
        losses: Dict[str, float]
            The dictionary holding the statistics of one step, such as the losses.
        phase: str
            Train or validation phase.
        epoch: int
        log_every_n_steps: int
            Every how many steps this function is called.
        global_step: int
        global_grad_step: int
        lr: float
            Current learning rate.
        beta: float
    """

    # logs = {"Step log ({}) {}".format(phase, stat_name, log_every_n_steps): v for stat_name, v in
    #         losses.items() if stat_name is not in ["iw_ll"]}

    logs = {}
    for stat_name, v in losses.items():
        # Those are only interesting to plot as histograms at the end of an epoch
        if stat_name not in ["iw_ll", "iw_ll_x_gen", "iw_ll_p_w", "iw_ll_x_gen_p_w", "lens", "lens_x_gen"]:
            logs["Step log ({}) {}".format(phase, stat_name, log_every_n_steps)] = v

    logs["epoch"] = epoch
    logs["global step"] = global_step
    logs["global gradient step"] = global_grad_step
    wandb.log(logs)


def insert_stats(stats, new_stats, epoch, phase):
    """
    Add losses to the statistics.

    Args:
        stats: defaultDict
            Nested dictionary with all the statistics so far.
        new_stats: Dict[str, float]
            Dictionary with new stats to insert in all the stats.
        epoch: int
        phase: str
            Train or validation

    Returns:
        stats: defaultDict
            Nested dictionary with all the statistics plus the ones added.
    """

    for stat_name, value in new_stats.items():
        # Make a list to save values to if first iteration of epoch
        if type(stats[epoch][phase][stat_name]) != list:
            stats[epoch][phase][stat_name] = []

        if torch.is_tensor(value):
            if stat_name in ["iw_ll", "iw_ll_x_gen", "iw_ll_p_w", "iw_ll_x_gen_p_w", "lens", "lens_x_gen"]:
                stats[epoch][phase][stat_name].extend(value.cpu().tolist())
            elif value.dim() > 0:
                stats[epoch][phase][stat_name].append(value.cpu().mean().item())
            else:
                stats[epoch][phase][stat_name].append(value.cpu().item())
        else:
            stats[epoch][phase][stat_name].append(value)

    return stats


# ----------------------------------------------------------------------------------------------------
# HELPER FUNCS
# ----------------------------------------------------------------------------------------------------
def get_code_dir():
    """
    Get the absolute path to the current working directory (NewsVAE directory).
    This differs per machine.

    Returns:
        code_dir_path: str
    """
    code_dir_path = os.path.dirname(os.path.realpath(__file__))

    return code_dir_path


def get_lr(scheduler):
    """
    Return the current learning rate, given the scheduler.

    Args:
        scheduler

    Returns:
        lr: float
            The current learning rate.
    """

    lr = scheduler.get_last_lr()[0]

    return lr


def print_stats(stats, epoch, phase, global_step, max_global_train_steps,
                global_grad_step, max_global_grad_train_steps,
                batch_i, phase_max_steps, objective="beta-vae"):
    """
    Print statistics to track process.

    Args:
        stats: defaultDict
            Nested dict with all stats in it.
        epoch: int
        phase: str
        global_step: int
        max_global_train_steps: int
        global_grad_step: int
        max_global_grad_train_steps: int
        batch_i: int
        phase_max_steps: int
        beta: float
        lr: float
    """
    print_string = "EPOCH {:4} | STEP {:5}/{:5} | GRAD STEP {:5}/{:5} | {:5} {:4}/{:4}".format(
        epoch, global_step + 1, max_global_train_steps, global_grad_step + 1,
        max_global_grad_train_steps, phase, batch_i + 1, phase_max_steps)

    print_string += "\n** GENERAL STATS"
    stat_dict = stats[epoch][phase]
    for s, v in stat_dict.items():
        if s != "LR":
            try:
                print_string += " | {}: {:8.2f}".format(s, v[-1])
            except:
                print("print error: ", s, v)
        else:
            if type(v[-1]) == float:
                print_string += " | {}: {:8.2f}".format(s, v[-1])

    # for s, v in stat_dict.items():
    #     if s not in ["alpha_MI", "beta_TC", "gamma_dim_KL", "alpha", "beta",
    #                  "gamma", "beta_KL", "KL", "TC", "MI", "dim_KL", "beta_marg_KL", 'marginal K']:
    #         if s != "LR":
    #             print_string += " | {}: {:8.2f}".format(s, v[-1])
    #         else:
    #             print_string += " | {}: {:8.8f}".format(s, v[-1])
    #
    #
    # # Beta-VAE
    # if objective == "beta-vae" or objective == "free-bits-beta-vae":
    #     print_string += f"\n** BETA-VAE | beta {stat_dict['beta'][-1]:.2f} x KL {stat_dict['KL'][-1]:.2f} = {stat_dict['beta_KL'][-1]:.2f}"
    #
    # # Beta-TC-VAE
    # elif objective == "beta-tc-vae":
    #     print_string += f"\n** BETA-TC-VAE | alpha {stat_dict['alpha'][-1]:.2f} x MI {stat_dict['MI'][-1]:.2f} = {stat_dict['alpha_MI'][-1]:.2f}"
    #     print_string += f" | beta {stat_dict['beta'][-1]:.2f} x TC {stat_dict['TC'][-1]:.2f} = {stat_dict['beta_TC'][-1]:.2f}"
    #     print_string += f" | gamma {stat_dict['gamma'][-1]:.2f} x Dim. KL {stat_dict['dim_KL'][-1]:.2f} = {stat_dict['gamma_dim_KL'][-1]:.2f}"
    #
    # elif objective == "hoffman":
    #     print_string += f"\n** HOFFMAN-VAE | alpha {stat_dict['alpha'][-1]:.2f} x MI {stat_dict['MI'][-1]:.2f} = {stat_dict['alpha_MI'][-1]:.2f}"
    #     print_string += f" | beta {stat_dict['beta'][-1]:.2f} x marg. KL {stat_dict['marginal KL'][-1]:.2f} = {stat_dict['beta_marg_KL'][-1]:.2f}"
    #
    print(print_string)


def get_number_of_params(model, trainable=True):
    """
    Count the number of (trainable) parameters in a model.

    Args:
        model: torch.nn.module
        trainable: bool:
            Whether to include trainable parameters only.
    Returns:
        n_params: int
    """

    if trainable:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n_params = sum(p.numel() for p in model.parameters())

    return n_params


def print_platform_code_dir():
    """
    Print info on the platform, node and code working directory.
    """

    platform_name, node = get_platform()
    print("Detected platform: {} ({})".format(platform_name, node))
    print("Code directory absolute path: {}".format(get_code_dir()))


def get_available_devices():
    """
    Determine which devices are available: GPU(s) or CPU only.

    Return:
        device: torch.device
        device_ids: list[int]
            a list of available GPU ids
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        print('{} GPU(s) detected'.format(len(device_ids)))
    else:
        device = torch.device("cpu")
        device_ids = []
        print('No GPU, available. Only CPU.')
    return device, device_ids


def make_nested_dict():
    """
    A function to initialise a nested default dict.
    """
    return collections.defaultdict(make_nested_dict)


def get_ip():
    """
    Returns the current IP.
    """
    IP = socket.gethostbyname(socket.gethostname())
    return IP


def transfer_batch_to_device(batch, device_name="cuda:0"):
    """
    Transfer an input batch to a device.

    Args:
        device_name: str:
            The name of the device to transfer the data to
        batch: Dict[str, Tensor]
            The data dict to put on device
    """
    for k in batch.keys():
        batch[k] = batch[k].to(device_name)
    return batch


def get_platform():
    """
    Deduce which platform we're on (lisa or local machine).

    Returns:
        platform_name: str:
            Name of the platform (lisa or local)
        platform: ... TODO: not sure what this is again

    :return:
    """
    platform_name = 'lisa' if 'lisa' in platform.node() else 'local'
    return platform_name, platform.node()


def cat_pad_uneven(list_of_blocks, pad_value=0):
    max_len = max([t.shape[1] for t in list_of_blocks])

    list_of_blocks_padded = [torch.nn.functional.pad(a, (0, max_len - a.shape[1]), mode='constant', value=pad_value)
                             for a in list_of_blocks]

    blocks_concatted = torch.cat(list_of_blocks_padded, dim=0)

    return blocks_concatted