from dataset.ski_dataloader import SkiDatamodule
import torch
import os
from absl import app
from absl import flags
import warnings
FLAGS = flags.FLAGS

# wandb.require("service")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def main(argv):
    init_all()
    train_dataloader = SkiDatamodule(FLAGS)
    train_features, train_labels = next(iter(train_dataloader))
    print('here ok')


if __name__ == '__main__':
    app.run(main)