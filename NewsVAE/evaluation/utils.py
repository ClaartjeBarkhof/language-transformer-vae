import sys; sys.path.append('../')
from train import get_model_on_device
from utils_train import load_from_checkpoint
import os
from pathlib import Path
import re
import pickle
from utils_train import transfer_batch_to_device
import sys

# Turn off ugly logging when working in notebooks
import logging
logging.disable(logging.WARNING)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_pickle(f):
    return pickle.load(open(f, "rb"))


def dump_pickle(o, f):
    pickle.dump(o, open(f, "wb"))


def get_PTB_runs(run_dir="/home/cbarkhof/code-thesis/NewsVAE/Runs", with_drop=False):
    dir_p = Path(run_dir)

    PTB_run_names_paths = {}
    for r in os.listdir(run_dir):
        if "PTB" in r:
            if "DROP" in r and with_drop is False:
                continue
            p = dir_p / r
            PTB_run_names_paths[r] = str(p)

    return PTB_run_names_paths


def get_clean_name(run_name):
    if "latent32" in run_name:
        nz = 32
    elif "latent64" in run_name:
        nz = 64
    else:
        nz = 768

    if "autoencoder" in run_name:
        fb = "autoencoder"
    else:
        fb = run_name[run_name.index("FB-")+3:].split("-")[0]
        if len(fb) == 3:
            fb += "0"

    if "emb" in run_name or "Emb" in run_name:
        mech = "mem+emb"
    else:
        mech = "mem"

    r_string = f"NZ-[{nz}] | FB-[{fb}] | MECH-[{mech}]"

    if "DROP" in run_name:
        drop = run_name.split('-')[5]
        r_string += f" | DROP[{drop}]"
    else:
        r_string += f" | DROP[{00}]"

    return r_string


def extract_info_from_clean_name(clean_run_name):
    res = re.split("\[|\]", clean_run_name)

    nz = int(res[1])
    fb = "autoencoder" if res[3] == "autoencoder" else float(res[3])
    mem = res[5]
    drop = int(res[7])

    # print(res)

    return nz, fb, mem, drop


def counted_validation_iterator(validation_loader):
    for batch_i, batch in enumerate(validation_loader):
        batch = transfer_batch_to_device(batch)
        yield batch_i, batch


def load_model_for_eval(run_name, path, device="cuda:0"):
    with HiddenPrints():

        if "checkpoint" not in path:
            path += "/checkpoint-best.pth"

        nz, _, mech, _ = extract_info_from_clean_name(get_clean_name(run_name))

        if "emb" in mech:
            add_latent_via_embeddings = True
        else:
            add_latent_via_embeddings = False

        if "mem" in mech:
            add_latent_via_memory = True
        else:
            add_latent_via_memory = False

        vae_model = get_model_on_device(device_name=device,
                                        latent_size=nz,
                                        gradient_checkpointing=False,
                                        add_latent_via_memory=add_latent_via_memory,
                                        add_latent_via_embeddings=add_latent_via_embeddings,
                                        do_tie_weights=True,
                                        do_tie_embedding_spaces=True,
                                        world_master=True,
                                        add_decoder_output_embedding_bias=False)

        _, _, vae_model, _, _, _, _ = load_from_checkpoint(vae_model, path)

        vae_model = vae_model.eval()

    return vae_model
