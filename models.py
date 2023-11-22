import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet, resnet50, resnet18
from torchvision.models.resnet import Bottleneck

__all__ = ['resnet50',
           'resnet18',
           'ResNet50Blocks',
           'ResNet18Blocks',
           'ResNet50BlocksPipeline',
           'ResNet18BlocksPipeline',
           'Resnet50_BLL_3splits'
           ]


class Resnet50_BLL_3splits(ResNet):
    '''
    This is a BLL Resnet50 model with splits manually inserted to divide the blocks, We found this to be the fastes to train.
    ( this model can also be JIT compiled)

    Inherits from original resnet model
    '''
    def __init__(self, no_detach=False, **kwargs):
        self.num_splits = 3
        self.no_detach = no_detach
        self.num_classes = kwargs['num_classes']
        super().__init__(Bottleneck, [3, 4, 6, 3],
                         num_classes=self.num_classes)

        self.bwd_pred_net = None
        self.bwd_net = None

    @property
    def num_blocks(self):
        return 4

    def create_bwd_net(self, sample_x_input):
        x = sample_x_input
        bwd_nets = []
        bwd_pred_nets = []
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)

            self.append_bwd_layer(x, bwd_nets, bwd_pred_nets)
            x = self.layer2(x)
            self.append_bwd_layer(x, bwd_nets, bwd_pred_nets)
            x = self.layer3(x)
            self.append_bwd_layer(x, bwd_nets, bwd_pred_nets)
        self.bwd_net = nn.ModuleList(bwd_nets)
        self.bwd_pred_net = nn.ModuleList(bwd_pred_nets)

    def append_bwd_layer(self, x, bwd_nets, bwd_pred_nets):
        bwd_size = (x.data.size(1))
        bwd_nets.append(nn.Sequential(
            nn.Linear(in_features=self.num_classes,
                      out_features=bwd_size, bias=False),
        )
        )
        c, h, w = nn.AvgPool2d(3, stride=2)(x).data.shape[1:]
        bwd_pred_nets.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(h),
            nn.Flatten(1),
            nn.Linear(in_features=c * h * w,
                      out_features=self.num_classes, bias=False)
        )
        )

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        activations.append(x)
        x = x.detach() if not self.no_detach else x

        x = self.layer2(x)
        activations.append(x)
        x = x.detach() if not self.no_detach else x

        x = self.layer3(x)
        activations.append(x)
        x = x.detach() if not self.no_detach else x

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # activations = [x]

        return x, activations, None


class Resnet_BLL(nn.Module):
    '''
    This is a BLL Resnet50 model with splits automatically inserted to divide the blocks, This is the model used in the experiments presented in the paper.

    Inherits from pytorch module and takes layers from `base_model`
    '''
    def __init__(self, base_model: 'resnet50', num_splits=3, no_detach=False, denoise=0., **kwargs):
        super().__init__()
        self.num_splits = num_splits
        self.no_detach = no_detach # Set `no_detach` to True in order to use vanilla reset
        self.denoise = denoise
        self.num_classes = kwargs['num_classes']
        rn50 = getattr(models, base_model)(num_classes=self.num_classes)
        head = nn.Sequential(rn50.conv1, rn50.bn1, rn50.relu, rn50.maxpool)
        tail = nn.Sequential(rn50.avgpool, nn.Flatten(1), rn50.fc)
        blocks = [head, *rn50.layer1, *rn50.layer2,
                  *rn50.layer3, *rn50.layer4, tail]
        self.blocks = nn.ModuleList(self._make_splits(blocks))

        self.bwd_pred_net = None
        self.bwd_net = None

    @property
    def num_blocks(self):
        return len(self.blocks)

    def _make_splits(self, blocks):
        if self.num_splits is None or self.num_splits == 0:
            return [nn.Sequential(*blocks)]
        else:
            new_blocks = []
            splits = self.num_splits + 1
            start_idx = 0

            num_blocks = len(blocks)
            inc = round(num_blocks / splits)

            while splits > 1:
                if splits == self.num_splits + 1:
                    new_blocks.append(nn.Sequential(
                        *blocks[start_idx: start_idx + inc]))
                else:
                    new_blocks.append(nn.Sequential(
                        *blocks[start_idx: start_idx + inc]))

                start_idx += inc
                splits -= 1

            new_blocks.append(nn.Sequential(*blocks[start_idx: num_blocks]))

            return new_blocks

    def create_bwd_net(self, sample_x_input):
        x = sample_x_input
        bwd_nets = []
        bwd_pred_nets = []
        with torch.no_grad():
            for block in self.blocks[:-1]:
                x = block(x)
                bwd_size = (x.data.size(1))
                bwd_nets.append(nn.Sequential(
                    nn.Linear(in_features=self.num_classes,
                              out_features=bwd_size, bias=False),
                )
                )
                c, h, w = nn.AvgPool2d(3, stride=2)(x).data.shape[1:]
                bwd_pred_nets.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(h),
                    nn.Flatten(1),
                    nn.Linear(in_features=c * h * w,
                              out_features=self.num_classes, bias=False)
                )
                )
        self.bwd_net = nn.ModuleList(bwd_nets)
        self.bwd_pred_net = nn.ModuleList(bwd_pred_nets)

    def forward(self, x):
        """
        Forward pass logic.
        """
        activations = []
        denoise_activations = []
        for i, block in enumerate(self.blocks):
            x_in = x
            x = block(x_in)
            activations.append(x)
            if self.denoise > 0:
                x_denoise = block(x_in + self.denoise *
                                  torch.randn_like(x_in))
                denoise_activations.append(x_denoise)
            x = x.detach() if not self.no_detach else x
        return activations[-1], activations[:-1], denoise_activations


class ResNetPiplined(Resnet_BLL):
    '''
    This is a BLL Resnet50 model pipelined implementation. Blocks are moved to different devices.
    Trining has to be slightly modified to compute posterior bootstrapping and back prop on relevant devices.

    Inherits from Resnet_BLL
    '''
    def __init__(self, *args, **kwargs):
        self.blocks = None
        self.bwd_pred_net = None
        self.bwd_net = None
        super().__init__(*args, **kwargs)

    def pipeline(self, first_dev):
        self.dev0 = first_dev
        print('moving model to devices:', [
              f'cuda:{first_dev + i}' for i in range(self.num_blocks)])
        for i, block in enumerate(self.blocks[:-1]):
            device = f'cuda:{first_dev + i}'
            block.to(device)
            self.bwd_net[i].to(device)
            self.bwd_pred_net[i].to(device)
        self.blocks[-1].to(f'cuda:{first_dev + self.num_blocks -1}')

    def forward(self, x):
        """
        Forward pass logic.
        """
        activations = [x]
        denoise_activations = [x]
        for i, block in enumerate(self.blocks):
            device = f'cuda:{self.dev0 + i}'
            x_in = x.to(device)
            x = block(x_in)
            activations.append(x)
            if self.denoise > 0:
                x_denoise = block(x_in + self.denoise *
                                  torch.randn_like(x_in))
                denoise_activations.append(x_denoise)
            x = x.detach() if not self.no_detach else x
        return activations[-1], activations, denoise_activations


def ResNet50Blocks(*args, **kwargs):
    return Resnet_BLL('resnet50', *args, **kwargs)


def ResNet18Blocks(*args, **kwargs):
    return Resnet_BLL('resnet18', *args, **kwargs)


def ResNet50BlocksPipeline(*args, **kwargs):
    return ResNetPiplined('resnet50', *args, **kwargs)


def ResNet18BlocksPipeline(*args, **kwargs):
    return ResNetPiplined('resnet18', *args, **kwargs)


def ResNet50Blocks_opt(*args, **kwargs):
    return Resnet50_BLL_3splits('resnet50', *args, **kwargs)
