import pytorch_lightning as pl
from typing import List, Dict
import torch
from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import NewsVAEArguments
from NewsData import NewsData
from typing import Tuple
import argparse


class NewsVAE(pl.LightningModule):
    """
    This Pytorch Lightning Module serves as a wrapper for training the EncoderDecoderShareVAE,
    which is a VAE composed of two checkpoints connected by a latent space.
    """

    def __init__(self, args: argparse.Namespace, roberta_ckpt_name: str = "roberta-base"):
        super().__init__()
        self.args = args

        # Encoder Decoder model
        self.encoder_decoder = EncoderDecoderShareVAE(args, roberta_ckpt_name)

    def training_step(self, article_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        losses = self.encoder_decoder(input_ids=article_batch['input_ids'],
                                      attention_mask=article_batch['attention_mask'],
                                      args=self.args)
        train_losses = {'train_'+k: v for k, v in losses.items()}
        return {'loss': losses['total_loss'], 'logs': train_losses}

    def validation_step(self, article_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        losses = self.encoder_decoder(input_ids=article_batch['input_ids'],
                                      attention_mask=article_batch['attention_mask'],
                                      args=self.args)
        valid_losses = {'valid_'+k: v for k, v in losses.items()}
        self.log_dict(valid_losses)
        return valid_losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=False,
                       using_lbfgs=False):
        # Linear warm-up
        if self.trainer.global_step < self.args.linear_lr_warmup_n_steps:
            lrs = torch.linspace(1e-5, args.learning_rate, args.linear_lr_warmup_n_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lrs[self.trainer.global_step]
        # Square root decay afterwards, not sure if this is the right implementation
        # https://stats.stackexchange.com/questions/200063/adam-optimizer-with-exponential-decay
        else:
            # Or this? https://fairseq.readthedocs.io/en/latest/lr_scheduler.html#fairseq.optim.lr_scheduler.inverse_square_root_schedule.InverseSquareRootSchedule
            # decay_factor = args.lr * torch.sqrt(args.warmup_updates)
            # lr = decay_factor / sqrt(update_num)

            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate / torch.sqrt(self.trainer.global_step)

        # update params
        optimizer.step()
        optimizer.zero_grad()


def main(args):
    news_data = NewsData(args.dataset_name, args.tokenizer_name,
                         batch_size=args.batch_size, num_workers=args.num_workers)
    with_PL = False

    if with_PL:
        news_vae = NewsVAE(args, 'roberta-base')
        trainer = pl.Trainer(accumulate_grad_batches=args.accumulate_n_batches_grad)
        trainer.fit(news_vae, news_data.dataloaders['train'], news_data.dataloaders['validation'])

    else:
        model = EncoderDecoderShareVAE(args, 'roberta-base')

        for batch in news_data.dataloaders['train']:
            kl_loss, recon_loss, total_loss = model(input_ids=batch['input_ids'],
                                                    attention_mask=batch['attention_mask'], args=args)
            print("kl_loss", kl_loss)
            print("recon_loss", recon_loss)
            print("total_loss", total_loss)
            break


if __name__ == "__main__":
    args = NewsVAEArguments.preprare_parser()
    main(args)
