import pytorch_lightning as pl
from torch import optim
from pysegmentation.models_unet import UNET_1D
import torch.nn.functional as Fun


class LitUnet(pl.LightningModule):
    def __init__(self, num_features=8,kernel_size=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNET_1D(input_dim=1,output_dim=5,
                             num_features=num_features,
                             kernel_size=kernel_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--num_features", type=int, default=8)
        parser.add_argument("--kernel_size", type=int, default=3)
        return parent_parser

    def step(self,batch,batch_idx,log):
        x, y = batch
        y_pred = self.model(x)
        loss = Fun.cross_entropy(y_pred, y)
        self.log(log, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"training_loss")
    def validation_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"validation_loss")
    def test_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"test_loss")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
