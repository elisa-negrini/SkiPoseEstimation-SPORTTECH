from matplotlib import use
import torch
import torch.nn as nn
import numpy as np
from residual_block import res_block

class Discriminator(nn.Module):
    def __init__(self, joint_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(joint_shape)), 1024),
            nn.LeakyReLU(0.2,inplace=True),
            res_block(num_neurons = 1024, use_batchnorm=False),
            nn.LeakyReLU(0.2,inplace=True),
            res_block(num_neurons=1024, use_batchnorm=False),
            nn.LeakyReLU(0.2,inplace=True),
            res_block(num_neurons=1024, use_batchnorm=False),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, joints):
        joints_flat = joints.view(joints.size(0), -1)
        validity = self.model(joints_flat)
        return validity