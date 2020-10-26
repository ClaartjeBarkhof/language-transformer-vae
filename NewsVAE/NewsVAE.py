import pytorch_lightning as pl
from typing import List, Dict
import torch
from EncoderDecoderShareVAE import EncoderDecoderShareVAE
import NewsVAEArguments
from NewsData import NewsData
from typing import Tuple


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

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        # This step should implement inference (maybe latent -> sample?)
        print("WARNING, IN FORWARD!")
        kl_loss, recon_loss = self.encoder_decoder(*batch_inputs)
        return kl_loss, recon_loss

    def training_step(self, article_batch: List[str], batch_idx: int) -> Dict:
        kl_loss, recon_loss = self.encoder_decoder(*article_batch)
        loss = kl_loss + recon_loss  # TODO: add beta term
        logs = {'loss': loss, 'kl_loss': kl_loss, 'recon_loss': recon_loss}
        return {'loss': loss, 'logs': logs}


def main(args):
    news_data = NewsData(args.dataset_name, args.tokenizer_name,
                         batch_size=args.batch_size, num_workers=args.num_workers)
    # news_vae = NewsVAE()
    #
    # trainer = pl.Trainer()
    # trainer.fit(news_vae, news_data.dataloaders['train'])  # TODO add val_step to pass , news_data.dataloaders['validation']

    model = EncoderDecoderShareVAE('roberta-base')

    for batch in news_data.dataloaders['train']:
        kl_loss, recon_loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        break

if __name__ == "__main__":
    args = NewsVAEArguments.preprare_parser()
    main(args)



