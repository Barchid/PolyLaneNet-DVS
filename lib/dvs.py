import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from . import sew_resnet
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from einops import rearrange

TIMESTEPS = 12
IN_CHANNELS = 2
MODE = "SNN"


def get_encoder_3d(in_channels: int) -> nn.Module:
    # resnet18 = models.video.r3d_18()
    resnet18 = models.video.mc3_18()
    resnet18.fc = nn.Identity()
    resnet18.stem[0] = nn.Conv3d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    # resnet18 = models.video.r2plus1d_18()
    # resnet18.fc = nn.Identity()
    # resnet18.stem[0] = nn.Conv3d(in_channels=in_channels, out_channels=45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
    return resnet18


def get_encoder(in_channels: int) -> nn.Module:
    resnet18 = models.resnet18(progress=True)

    resnet18.fc = nn.Identity()

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


def get_encoder_snn(in_channels: int, T: int, output_all: bool):
    resnet18 = sew_resnet.MultiStepSEWResNet(
        block=sew_resnet.MultiStepBasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        T=T,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepIFNode,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
        output_all=output_all,
    )

    # resnet18.layer4[-1].sn2 = MultiStepLIAFNode(
    #     torch.nn.ReLU(),
    #     threshold_related=False,
    #     detach_reset=True,
    #     surrogate_function=surrogate.ATan(),
    # )

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


class SNNModule(nn.Module):
    """Some Information about SNNModule"""

    def __init__(self, mode=MODE, n_classes=10):
        super(SNNModule, self).__init__()

        if mode == "3dcnn":
            self.encoder = get_encoder_3d(in_channels=IN_CHANNELS)
        elif mode == "snn":
            self.encoder = get_encoder_snn(
                in_channels=IN_CHANNELS, T=TIMESTEPS, output_all=False
            )
        else:  # cnn
            self.encoder = get_encoder(in_channels=IN_CHANNELS * TIMESTEPS)

        self.fc = nn.Linear(512, n_classes, bias=False)

        self.mode = mode

    def forward(self, x):
        # x = B,3,H,W
        x = x[:, 0:2, :, :]  # x = B,2,H,W
        x.unsqueeze_(0)
        x = x.repeat(TIMESTEPS, 1, 1, 1, 1)  # x = T,B,2,H,W
        
        if self.mode == "snn":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)
        elif self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)
        elif self.mode == "cnn" and len(x.shape) == 5:
            x = rearrange(
                x,
                "batch time channel height width -> batch (time channel) height width",
            )

        x = self.encoder(x)
        x = self.fc(x)
        return x
