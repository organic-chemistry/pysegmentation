from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
from torch import optim, nn, utils, Tensor



from pysegmentation.load_data import load_data_ml
from pysegmentation.models_light import LitUnet
from pysegmentation.datasets import Dataset1dto1d



parser = ArgumentParser()
parser.add_argument("--data", type=str, nargs='+',default=[])
parser.add_argument("--bin_size", type=int,default=200)
parser.add_argument("--read_size", type=int,default=256)

parser = pl.Trainer.add_argparse_args(parser)

parser = LitUnet.add_model_specific_args(parser)

args = parser.parse_args()
#print(args)

X=[]
Y=[]
for name in args.data:
    x,y = load_data_ml(root=name,bin_size=args.bin_size,read_size=args.read_size)
    X.extend(x)
    Y.extend(y)


md = Dataset1dto1d(X,Y)

#Split the data

import torch.utils.data as data
train_set_size = int(len(md) * 0.9)
valid_set_size = len(md) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(md, [train_set_size, valid_set_size], generator=seed)

train_loader = utils.data.DataLoader(train_set,batch_size=8)
val_loader = utils.data.DataLoader(valid_set,batch_size=8)

dict_args = vars(args)

model = LitUnet(dict_args["num_features"],dict_args["kernel_size"])

trainer =  pl.Trainer.from_argparse_args(args, callbacks=[EarlyStopping(monitor="validation_loss", mode="min",patience=10)])


trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)
