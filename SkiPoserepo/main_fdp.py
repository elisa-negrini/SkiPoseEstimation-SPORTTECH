from distutils.command.config import config
from datamodule_ski import SkiDataModule
from datamodule_body_25 import Body25DataModule
from model import AdaptationNetwork
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
import numpy as np
import domainadapt_flags
from absl import app
from absl import flags
import warnings
FLAGS = flags.FLAGS

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

def init_all():
    warnings.filterwarnings("ignore")

    # enable cudnn and its inbuilt auto-tuner to find the best algorithm to use for your hardware
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # useful for run-time
    torch.backends.cudnn.deterministic = True

    # pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()

# def sweep_iteration():
    

def main(argv):
    init_all()
    wandb.init(project="DAM", name="from %s to %s" %(FLAGS.dataset, FLAGS.train_dataset))
    wandb_logger = WandbLogger()
    config = wandb.config
    if FLAGS.mode == "train":
        model = AdaptationNetwork(FLAGS)
        if FLAGS.fine_tuning:
            ckpt = torch.load(FLAGS.load_pretrained_model) 
            model.load_state_dict(ckpt["state_dict"])
        if FLAGS.dataset == 'ski':
            dm = SkiDataModule(FLAGS)
        directory = "/home/federico.diprima/DAT/DAMski/models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        trainer = Trainer(
            default_root_dir=directory,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            #track_grad_norm=2,
            detect_anomaly=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],logger=wandb_logger)
        trainer.fit(model, dm)
        print("\n\nFind the visual results in", FLAGS.train_result_dir)
        trainer.save_checkpoint(FLAGS.load_checkpoint)
        print("\n\nFind the model checkpoint in", FLAGS.load_checkpoint)

    if FLAGS.mode == "demo":
        model = AdaptationNetwork(FLAGS)
        if FLAGS.dataset == 'body_25':
            dm = Body25DataModule(FLAGS)
        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            #track_grad_norm=2,
            detect_anomaly=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],logger=wandb_logger)
        trainer.test(model = model,dataloaders = dm,ckpt_path=FLAGS.load_checkpoint)
        print("\n\nFind the visual results in", FLAGS.test_result_dir)
    

if __name__ == '__main__':
    app.run(main)