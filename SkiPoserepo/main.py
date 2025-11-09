from dataset.ski_datamodule import SkiDatamodule
from model import SkiDat_Network
from pytorch_lightning import  Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import wandb
from absl import app
from absl import flags
import warnings
from domainadapt_flags import FLAGS


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
    wandb.init(project="skidat", name = "Image Patches L1" )#%(str(FLAGS.batch_size)))
    wandb_logger = WandbLogger()
    # config = wandb.config
    if FLAGS.mode == "train":
        dm = SkiDatamodule(FLAGS)
        model = SkiDat_Network(FLAGS)

        val_checkpoint = ModelCheckpoint(monitor="Validation/loss", mode="min")

        trainer = Trainer(
            gpus=1,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            # track_grad_norm=2,
            detect_anomaly=False,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=20)],logger=wandb_logger)

        trainer.fit(model, dm)
    

if __name__ == '__main__':
    app.run(main)