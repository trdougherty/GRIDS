import argparse
from os.path import exists
import yaml
import numpy as np
import wandb
import collections

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# The local files for ml
from ml.data_split import DataSplit
from ml.model_loader import ModelLoader

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, action="store", dest="lr")
parser.add_argument("--batch_size", type=int, action="store", dest="batch_size")
parser.add_argument("--epochs", type=int, action="store", dest="epochs")
parser.add_argument("--note", action="store", dest="note", default="")
parser.add_argument("--settings", action="store", dest="settings", default="training_settings.txt")
parser.add_argument("-tr", dest="train", action='store_true', help="Train the model", default=False)
parser.add_argument("-tr_ts", dest="train_test", action='store_true', help="Test training the model",default=False)
parser.add_argument("-ts", dest="test", action='store_true', help="test the pretrained model", default=False)
parser.add_argument('-ckpt', action="store", dest="ckpt")
parser.add_argument('-yaml', action="store", dest="yaml")
args = parser.parse_args()

# If we have a file which is running the show, 
config = {}

# And then the command line arguments will get the last say
config = { **config, **vars(args) }

if exists(args.settings):
    with open(args.settings, 'r') as f:
        settings = yaml.safe_load(f)
    config = {**config, **settings}

# config['name'] = "dbr-{}-{}".format(config['lr'], config['batch_size'])
wandb.init(config=config)

# To try and make the experiments reproducible
seed = 42
np.random.seed(seed)
pl.seed_everything(seed)

wandb_logger = WandbLogger(name='001', project='dbr')

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    filename='{epoch:02d}-{val_loss:.2f}',
    auto_insert_metric_name=True
)

trainer = pl.Trainer(
    gpus=config['gpus'],
    logger=wandb_logger,
    max_epochs=config['epochs'],
    callbacks=[checkpoint_callback]
)

data_module = DataSplit(**config)

if args.train:
    model = ModelLoader(input_size=1007)
    trainer.fit(model, data_module)

if args.test:
    model = ModelLoader.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        hparams_file=args.yaml ,
        map_location=None
    )
    trainer.test(model, data_module)
    