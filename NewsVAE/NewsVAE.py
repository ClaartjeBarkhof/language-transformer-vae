import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from typing import List, Dict, Any
import torch
from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import NewsVAEArguments
from NewsData import NewsDataModule
import argparse
import utils
from datetime import datetime
import numpy as np


class NewsVAE(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, roberta_ckpt_name: str = "roberta-base", batch_size: int = 8):
        super().__init__()
        self.args = args
        self.hparams = args
        self.batch_size = batch_size

        # Encoder Decoder model
        self.encoder_decoder = EncoderDecoderShareVAE(args, roberta_ckpt_name, do_tie_weights=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):

        # Strange that this is needed
        batch = self.transfer_batch_to_device(batch, self.device)

        losses = self.encoder_decoder(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      args=self.args)

        train_losses = {'train_' + k: v for k, v in losses.items()}

        self.log_dict(train_losses, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return {'loss': losses['total_loss']}

    def validation_step(self, batch, batch_idx):
        # Strange that this is needed
        batch = self.transfer_batch_to_device(batch, self.device)

        losses = self.encoder_decoder(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      args=self.args)
        valid_losses = {'valid_' + k: v for k, v in losses.items()}

        self.log_dict(valid_losses, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return {'loss': losses['total_loss']}

    # def gather_outputs(self, outputs: List[Any], prefix) -> Dict[str, torch.Tensor]:
    #     print(len(outputs))
    #     epoch_stats = {}
    #     for loss_name in outputs[0].keys():
    #         stacked_losses = torch.stack([o[loss_name] for o in outputs])
    #         mean_loss = stacked_losses.mean()
    #         epoch_stats[prefix + loss_name] = mean_loss
    #     return epoch_stats
    #
    # def training_epoch_end(self, outputs: List[Any]) -> None:
    #     epoch_train_stats = self.gather_outputs(outputs, 'train_epoch_')
    #     self.log_dict(epoch_train_stats, logger=True)
    #
    # def validation_epoch_end(self, outputs: List[Any]) -> None:
    #     epoch_val_stats = self.gather_outputs(outputs, 'valid_epoch_')
    #     self.log_dict(epoch_val_stats, logger=True)

    # def backward(self, use_amp, loss, optimizer):
    #     """
    #     Override backward with your own implementation if you need to
    #     :param use_amp: Whether amp was requested or not
    #     :param loss: Loss is already scaled by accumulated grads
    #     :param optimizer: Current optimizer being used
    #     :return:
    #     """
    #     if use_amp:
    #         with amp.scale_loss(loss, optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         loss.backward()

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

    datetime_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = "experiment-{}".format(datetime_stamp)

    loggers = [pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=experiment_name),
               pl_loggers.WandbLogger(save_dir='lightning_logs', project='thesis', name=experiment_name)]

    news_data = NewsDataModule(args.dataset_name, args.tokenizer_name,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

    news_vae = NewsVAE(args, 'roberta-base', batch_size=args.batch_size,)

    trainer = pl.Trainer(logger=loggers,
                         fast_dev_run=args.fast_dev_run,
                         accumulate_grad_batches=args.accumulate_n_batches_grad,
                         gpus=args.n_gpus,
                         max_steps=args.max_steps,
                         max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         distributed_backend=args.accelerator,
                         num_nodes=1,
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

    print("trainer.on_gpu", trainer.on_gpu)
    print("trainer.distributed_backend", trainer.distributed_backend)
    print("trainer.data_parallel_device_ids", trainer.data_parallel_device_ids)
    print("trainer.root_gpu", trainer.root_gpu)

    trainer.fit(news_vae, datamodule=news_data)


if __name__ == "__main__":
    config = NewsVAEArguments.preprare_parser()
    main(config)