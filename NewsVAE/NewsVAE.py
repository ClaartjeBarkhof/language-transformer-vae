import pytorch_lightning as pl
from typing import List, Dict
import torch
from torch.nn import functional as F
from torch import nn
from transformers import RobertaTokenizer, BertGenerationEncoder, BertGenerationDecoder
from EncoderDecoderShareVAE import EncoderDecoderShareVAE


class NewsVAE(pl.LightningModule):
    """
    This Pytorch Lightning Module serves as a wrapper for training the EncoderDecoderShareVAE,
    which is a VAE composed of two checkpoints connected by a latent space.
    """

    def __init__(self, roberta_ckpt_name: str = "roberta-base"):
        super().__init__()

        # Encoder Decoder model
        self.encoder_decoder = EncoderDecoderShareVAE(roberta_ckpt_name)

    def configure_optimizers(self):
        # TODO: this is just a placeholder simple Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def forward(self, article_batch: List[str]) -> Dict:
        batch_inputs = self.encoder_decoder.tokenizer(article_batch, padding=True, truncation=True)
        kl_loss, recon_loss = self.encoder_decoder(batch_inputs)
        return kl_loss, recon_loss

    def training_step(self, article_batch: List[str]) -> Dict:
        kl_loss, recon_loss = self(article_batch)
        loss = kl_loss + recon_loss  # TODO: add beta term
        logs = {'loss': loss, 'kl_loss': kl_loss, 'recon_loss': recon_loss}
        return {'loss': loss, 'logs': logs}


if __name__ == "__main__":
    NewsVAE()
