from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import torch
import numpy as np
import sys
import math
import NewsVAEArguments
from NewsData import NewsData
# from trainNewsVAE import load_from_checkpoint
import copy
import utils


def log_sum_exp(value, dim=None, keepdim=False):
    # Taken from: https://github.com/bohanli/vae-pretraining-encoder/blob/master/utils.py
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def calc_au(model, test_data_batch, delta=0.01):
    # Taken from: https://github.com/bohanli/vae-pretraining-encoder/blob/master/utils.py
    """Compute the number of active units"""
    cnt = 0
    for batch_data in test_data_batch:
        encoder_outs = model.encoder(input_ids=batch_data["input_ids"],
                                     attention_mask=batch_data["attention_mask"])  # CLAARTJE

        mean, _ = encoder_outs.pooler_output.chunk(2, dim=1)  # CLAARTJE

        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0

    for batch_data in test_data_batch:
        encoder_outs = model.encoder(input_ids=batch_data["input_ids"],
                                     attention_mask=batch_data["attention_mask"])  # CLAARTJE

        mean, _ = encoder_outs.pooler_output.chunk(2, dim=1)  # CLAARTJE
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


#
def calc_mi(model, test_data_batch, device="cuda:0"):
    # Taken from: https://github.com/bohanli/vae-pretraining-encoder/blob/master/utils.py
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.

    for batch_data in test_data_batch:
        # Forward the encoder
        encoder_outs = model.encoder(input_ids=batch_data["input_ids"],
                                     attention_mask=batch_data["attention_mask"])  # CLAARTJE

        mu, logvar = encoder_outs.pooler_output.chunk(2, dim=1)  # CLAARTJE

        x_batch, nz = mu.size()
        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy += (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]

    neg_entropy = neg_entropy / num_examples

    num_examples = 0
    log_qz = 0.
    for i in range(len(mu_batch_list)):
        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()

        z_samples = model.reparameterize(mu, logvar)  # CLAARTJE

        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        # mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        # indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi


def nll_iw(model, x, nsamples, args, ns=100):
    """compute the importance weighting estimate of the log-likelihood
    Args:
        x: if the data is constant-length, x is the data tensor with
            shape (batch, *). Otherwise x is a tuple that contains
            the data tensor and length list
        nsamples: Int
            the number of samples required to estimate marginal data likelihood
    Returns: Tensor1
        Tensor1: the estimate of log p(x), shape [batch]
    """

    # compute iw every ns samples to address the memory issue
    # nsamples = 500, ns = 100
    # nsamples = 500, ns = 10

    # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
    # .      this problem is to be solved in order to speed up

    tmp = []
    for _ in range(int(nsamples / ns)):
        # [batch, ns, nz]
        # param is the parameters required to evaluate q(z|x)

        # Make use of the pooled features to sample a latent z vector
        encoder_outs = model.encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        latent_z, param = model.sample(encoder_outs.pooler_output, ns)

        # [batch, ns]
        # eval_complete_ll(self, input_ids, attention_mask, latent_z, args)
        log_comp_ll = model.eval_complete_ll(x["input_ids"], x["attention_mask"], latent_z, args)
        log_infer_ll = model.eval_inference_dist(x["input_ids"], x["attention_mask"], latent_z, param)

        tmp.append(log_comp_ll - log_infer_ll)

    ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

    return -ll_iw


def calc_iwnll(model, test_data_batch, args, iw_nsamples, ns=100):
    report_nll_loss = 0
    report_num_words = report_num_sents = 0

    print("iw nll computing ", end="")

    for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data["input_ids"].size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size

        if id_ % (round(len(test_data_batch) / 20)) == 0:
            print('%d%% ' % (id_ / (round(len(test_data_batch) / 20)) * 5), end="")
            sys.stdout.flush()

        loss = nll_iw(model, batch_data, iw_nsamples, args, ns=ns)

        report_nll_loss += loss.sum().item()

    print()
    sys.stdout.flush()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    return nll, ppl


def batch_decode(batch_of_samples, tokenizer):
    return [tokenizer.decode(a.tolist()) for a in batch_of_samples]


def evaluate_model(VAE_model, data, args, n_batches=100, iw_nsamples=200):
    dataloader = data.val_dataloader(shuffle=True)

    text_results = {
        "input_text": [],
        "reconstruction_text": []}

    results = {
        "kl_loss": [],
        "mmd_loss": [],
        "recon_loss": [],
        "total_loss": [],
        "hinge_kl_loss": [],
        "exact_match_acc": [],
        "mutual_information": None,
        "active_units": None,
        "nll": None,
        "ppl": None
    }

    VAE_model.eval()

    # Get some data to evaluate on
    batches = []
    for batch_i, input_batch in enumerate(dataloader):

        batches.append(utils.transfer_batch_to_device(input_batch, 0))

        if batch_i - 1 == n_batches:
            break

    # Calculate some metrics on all batches at once
    with torch.no_grad():
        nll, ppl = calc_iwnll(VAE_model, batches, args, iw_nsamples)

        results['nll'] = nll
        results['ppl'] = ppl

        # Mutual information
        mi = calc_mi(VAE_model, batches)
        results["mutual_information"] = mi.item()

        # Active units
        au, _ = calc_au(VAE_model, batches)
        results["active_units"] = au

        # Get predictions / losses / reconstructions
        for batch_i, input_batch in enumerate(batches):
            predictions = VAE_model(input_ids=input_batch['input_ids'],
                                    attention_mask=input_batch['attention_mask'],
                                    beta=1.0, args=args, return_predictions=True,
                                    return_exact_match_acc=True)

            predictions['total_loss'] = predictions['total_loss'].item()

            for l_name, v in predictions.items():
                if "loss" in l_name or "acc" in l_name:
                    results[l_name].append(v)

            input_texts = batch_decode(input_batch['input_ids'], data.tokenizer)
            reconstructions = batch_decode(predictions['predictions'], data.tokenizer)

            text_results["input_text"].extend(input_texts)
            text_results["reconstruction_text"].extend(reconstructions)

    # Take the average over all batches
    for k, v in results.items():
        if type(v) == list:
            results[k] = np.mean(v)

    return text_results, results


if __name__ == "__main__":
    # Overwrite some arguments
    args = NewsVAEArguments.preprare_parser(jupyter=False, print_settings=False)
    args.ddp = False
    args.n_gpus = 1

    # Check if on server
    if torch.cuda.is_available():
        print("Using GPU :-)...")
        device = "cuda"
    else:
        print("Warning: using CPU!")
        device = "cpu"

    # Get data
    data = NewsData(args.dataset_name, args.tokenizer_name,
                    batch_size=args.batch_size, num_workers=4,
                    pin_memory=(device == "cuda"), debug=False,
                    debug_data_len=args.debug_data_len,
                    max_seq_len=args.max_seq_len, device=device)

    # Get model architecture
    VAE_model = EncoderDecoderShareVAE(args, args.base_checkpoint_name, do_tie_weights=args.do_tie_weights).to(device)
    text_results, results = evaluate_model(VAE_model, data, args, iw_nsamples=300, n_batches=12)

    print(text_results)
    print(results)
