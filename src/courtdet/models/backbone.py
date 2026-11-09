"""Convolutional backbone: ImageNet ResNet-18 truncated after `layer2` or `layer3`.

ResNet-18 is a stack of convolution blocks; each stage halves the resolution and
increases the number of channels:

    stage             stride  channels  theoretical receptive field
    conv1 + maxpool   4        64
    layer1            4        64        43 px
    layer2            8       128        99 px
    layer3            16      256       211 px
    layer4            32      512       435 px

`features` picks the map given to the RPN and the RoI head:

  "layer2" (default): (H/8, W/8) cells, 128 channels. A fine grid, which matters for
      small objects such as courts seen from above; but a cell sees about one court
      (median ~76-93 px), little of its surroundings. The original project made this
      choice (the notebook's comments say "layer3": they are wrong, the code stops after layer2).
  "layer3": (H/16, W/16), 256 channels. Richer, more "semantic" features with twice the
      receptive field, but a coarser grid: a 19 px court covers about one cell.
  "fused": layer3 brought back to the layer2 grid, as one level of a Feature Pyramid
      Network (Lin et al. 2017, "top-down pathway + lateral connections"):
          out = smooth( lateral2(layer2) + upsample_x2(lateral3(layer3)) )
      lateral2 / lateral3 are 1x1 convolutions that bring both maps to 128 channels, the
      upsampling is nearest-neighbour, and `smooth` (3x3) reduces the blockiness of the
      upsampling. Stride 8 and 128 channels, as "layer2": the anchors, the RPN and the RoI
      head do not change, only what each cell knows.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

FEATURES = ("layer2", "layer3", "fused")
_LAST_STAGE = {"layer2": "layer2", "layer3": "layer3", "fused": "layer3"}
_OUT = {"layer2": (128, 8), "layer3": (256, 16), "fused": (128, 8)}  # (channels, stride)


class ResNet18Backbone(nn.Module):
    """ResNet-18 up to layer2 or layer3: [B, 3, H, W] -> [B, out_channels, H/stride, W/stride].

    Fine-tuning policy (the notebook's, made explicit):
      - everything is frozen except the convolution weights of `trainable_layers`
        (default: layer2). The early layers detect generic edges and textures that
        transfer from ImageNet to satellite images; the later ones adapt to the new domain.
        The FPN convolutions of "fused" are new, hence always trained.
      - BatchNorm layers are ALWAYS frozen: they keep their ImageNet statistics, even
        when `.train()` is called on a parent module. With batches of ~8 images, the
        statistics of a batch are noisy, and updating them would drift away from what
        the pretrained weights expect. (The notebook froze the statistics too, but
        kept training the BatchNorm scale/shift in layer2, and relied on calling
        `backbone.eval()` by hand at every epoch.)

    Args:
        pretrained: load ImageNet weights (downloaded on first use). Tests use False.
        trainable_layers: names of the ResNet stages whose convolutions are trained.
        features: "layer2", "layer3" or "fused" (see the module docstring).
    """

    def __init__(self, pretrained: bool = True, trainable_layers: Sequence[str] = ("layer2",),
                 features: str = "layer2"):
        super().__init__()
        if features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
        stages = ["layer1", "layer2", "layer3"][: 2 if _LAST_STAGE[features] == "layer2" else 3]
        unknown = set(trainable_layers) - set(stages)
        if unknown:  # e.g. layer3 trainable while features="layer2": the layer would not exist
            raise ValueError(f"trainable_layers {sorted(unknown)} are not part of the {features!r} backbone")
        self.features = features
        self.out_channels, self.stride = _OUT[features]

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        if "layer3" in stages:
            self.layer3 = resnet.layer3

        for name, param in self.named_parameters():
            param.requires_grad_(any(name.startswith(f"{layer}.") for layer in trainable_layers))
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad_(False)

        if features == "fused":  # created after the freezing loop: new layers, always trained
            self.lateral2 = nn.Conv2d(128, 128, kernel_size=1)
            self.lateral3 = nn.Conv2d(256, 128, kernel_size=1)
            self.smooth = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            for conv in (self.lateral2, self.lateral3, self.smooth):
                # torchvision's FPN initialisation (torchvision/ops/feature_pyramid_network.py)
                nn.init.kaiming_uniform_(conv.weight, a=1)
                nn.init.zeros_(conv.bias)
        self.train()  # applies the BatchNorm rule below

    def train(self, mode: bool = True) -> "ResNet18Backbone":
        """Same as nn.Module.train, except BatchNorm layers stay in eval mode."""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer2(self.layer1(x))
        if self.features == "layer2":
            return c2
        c3 = self.layer3(c2)
        if self.features == "layer3":
            return c3
        # size= rather than scale_factor=2: an odd c2 size (H/8 odd) gives c3 = ceil(c2 / 2),
        # and x2 would be one cell too large
        top_down = F.interpolate(self.lateral3(c3), size=c2.shape[-2:], mode="nearest")
        return self.smooth(self.lateral2(c2) + top_down)
