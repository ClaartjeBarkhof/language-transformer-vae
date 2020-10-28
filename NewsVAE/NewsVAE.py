import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from typing import List, Dict
import torch
from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import NewsVAEArguments
from NewsData import NewsDataModule
import argparse
import utils
import os
import numpy as np


class NewsVAE(pl.LightningModule):
    """
    This Pytorch Lightning Module serves as a wrapper for training the EncoderDecoderShareVAE,
    which is a VAE composed of two RoBERTa checkpoints connected by a through a latent space.
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
        train_losses = {'train_' + k: v for k, v in losses.items()}
        self.log_dict(train_losses, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return {'loss': losses['total_loss']}

    def validation_step(self, article_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        losses = self.encoder_decoder(input_ids=article_batch['input_ids'],
                                      attention_mask=article_batch['attention_mask'],
                                      args=self.args)
        valid_losses = {'valid_' + k: v for k, v in losses.items()}
        self.log_dict(valid_losses, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return valid_losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=False,
                       using_lbfgs=False):
        # Linear warm-up
        if self.trainer.global_step < self.args.linear_lr_warmup_n_steps:
            lrs = torch.linspace(1e-5, self.args.learning_rate, self.args.linear_lr_warmup_n_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lrs[self.trainer.global_step]

        # Square root decay afterwards, not sure if this is the right implementation
        # https://stats.stackexchange.com/questions/200063/adam-optimizer-with-exponential-decay
        else:
            # Or this? https://fairseq.readthedocs.io/en/latest/lr_scheduler.html#fairseq.optim.lr_scheduler.inverse_square_root_schedule.InverseSquareRootSchedule
            # decay_factor = args.lr * torch.sqrt(args.warmup_updates)
            # lr = decay_factor / sqrt(update_num)

            for pg in optimizer.param_groups:
                pg['lr'] = self.args.learning_rate / np.sqrt(self.trainer.global_step)

        # update params
        optimizer.step()
        optimizer.zero_grad()


def main(args):
    utils.print_platform_codedir()

    if args.deterministic:
        seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint()

    loggers = [pl_loggers.TensorBoardLogger(save_dir='lightning_logs'),
               pl_loggers.WandbLogger(save_dir='lightning_logs', project='thesis')]

    news_data = NewsDataModule(args.dataset_name, args.tokenizer_name,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

    news_vae = NewsVAE(args, 'roberta-base')

    trainer = pl.Trainer(logger=loggers,
                         accumulate_grad_batches=args.accumulate_n_batches_grad,
                         gpus=args.n_gpus,
                         max_steps=args.max_steps,
                         max_epochs=args.max_epochs,
                         accelerator=args.distributed_backend,
                         num_nodes=args.n_nodes,
                         auto_select_gpus=True,
                         benchmark=True,  # makes system faster with equal sized batches
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         checkpoint_callback=checkpoint_callback,
                         log_gpu_memory=args.log_gpu_memory,
                         log_every_n_steps=args.log_every_n_steps,
                         # sync_batchnorm=True,  # Do I want this?
                         track_grad_norm=True,
                         # weights_summary='full',
                         default_root_dir=utils.get_code_dir()
                         )

    trainer.fit(news_vae, news_data)


if __name__ == "__main__":
    config = NewsVAEArguments.preprare_parser()
    main(config)

    #     model = EncoderDecoderShareVAE(args, 'roberta-base')
    #
    #     for batch in news_data.dataloaders['train']:
    #         kl_loss, recon_loss, total_loss = model(input_ids=batch['input_ids'],
    #                                                 attention_mask=batch['attention_mask'], args=args)
    #         print("kl_loss", kl_loss)
    #         print("recon_loss", recon_loss)
    #         print("total_loss", total_loss)
    #         break
