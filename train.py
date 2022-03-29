import argparse
from os.path import exists
import numpy as np
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# The local files for ml
from src.ml.data_split import DataSplit
from src.ml.model_loader import ModelLoader

parser = argparse.ArgumentParser()
parser.add_argument("-tr", dest="train", action='store_true', help="Train the model",default=False)
parser.add_argument("-ts", dest="test", action='store_true', help="Test the trained model", default=False)
parser.add_argument('-settings', dest="settings", action='store_true', help="Name of settings file")
parser.add_argument('-ckpt', action="store", dest="ckpt")
parser.add_argument('-yaml', action="store", dest="yaml")
args = parser.parse_args()

logger = TensorBoardLogger("lightning_logs", name="model")
data_module = DataSplit()
trainer = pl.Trainer(gpus=0, logger=logger, max_epochs=10)

settings = None
if exists(args.settings):
    settings = yaml.safe_load(args.settings)

if args.train:
    model = ModelLoader()
    trainer.fit(model, data_module)

if args.test:
    model = ModelLoader.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        hparams_file=args.yaml ,
        map_location=None
    )
    trainer.test(model, data_module)
    