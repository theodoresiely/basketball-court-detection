import torch
from torch import nn

from courtdet.models.backbone import ResNet18Backbone


def make():
    # Random weights: no download, the tests stay offline. They check structure, not accuracy.
    return ResNet18Backbone(pretrained=False)


def test_output_shape_follows_stride_8():
    backbone = make()
    out = backbone(torch.randn(1, 3, 256, 384))  # non-square on purpose: catches H/W mix-ups
    assert out.shape == (1, 128, 32, 48)
    out = backbone(torch.randn(2, 3, 512, 512))  # the project's configuration
    assert out.shape == (2, 128, 64, 64)


def test_only_layer2_convolutions_are_trainable():
    backbone = make()
    trainable = {n: p for n, p in backbone.named_parameters() if p.requires_grad}
    # layer2 = 2 residual blocks with 2 convs each, plus 1 downsampling conv in the first block
    assert len(trainable) == 5
    for name in trainable:
        assert name.startswith("layer2.")
        assert isinstance(backbone.get_submodule(name.rsplit(".", 1)[0]), nn.Conv2d)  # no BatchNorm
    # hand count: 64*128*9 (first conv) + 3 * 128*128*9 + 64*128*1 (downsampling) = 524,288
    assert sum(p.numel() for p in trainable.values()) == 524_288


def test_batchnorm_stays_frozen_in_train_mode():
    backbone = make()
    backbone.train()
    bns = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    before = backbone.bn1.running_mean.clone()
    backbone(torch.randn(2, 3, 64, 64) + 5.0)  # input far from the stored statistics
    assert torch.equal(backbone.bn1.running_mean, before)  # a BatchNorm in train mode would move
    assert backbone.layer2[0].conv1.training  # the convolutions do follow the mode


def test_gradients_only_reach_layer2():
    backbone = make()
    backbone(torch.randn(1, 3, 64, 64)).sum().backward()
    assert backbone.layer2[0].conv1.weight.grad is not None
    assert backbone.layer1[0].conv1.weight.grad is None
    assert backbone.conv1.weight.grad is None


def test_layer3_and_fused_shapes():
    # layer3: twice coarser, 256 channels; fused: layer3 brought back to the layer2 grid
    b3 = ResNet18Backbone(pretrained=False, trainable_layers=("layer2", "layer3"), features="layer3")
    assert (b3.out_channels, b3.stride) == (256, 16)
    assert b3(torch.randn(1, 3, 256, 384)).shape == (1, 256, 16, 24)
    bf = ResNet18Backbone(pretrained=False, trainable_layers=("layer2", "layer3"), features="fused")
    assert (bf.out_channels, bf.stride) == (128, 8)
    assert bf(torch.randn(1, 3, 256, 384)).shape == (1, 128, 32, 48)
    assert bf(torch.randn(1, 3, 72, 72)).shape == (1, 128, 9, 9)  # odd layer2 grid: upsampling by size, not x2


def test_layer2_backbone_has_no_layer3_and_rejects_it_as_trainable():
    assert not hasattr(make(), "layer3")  # old checkpoints: same state_dict keys as before
    try:
        ResNet18Backbone(pretrained=False, trainable_layers=("layer3",), features="layer2")
    except ValueError:
        pass
    else:
        raise AssertionError("layer3 cannot be trainable when the backbone stops at layer2")


def test_fused_trains_the_fpn_convs_and_layer3_but_not_the_frozen_stages():
    bf = ResNet18Backbone(pretrained=False, trainable_layers=("layer2", "layer3"), features="fused")
    bf(torch.randn(1, 3, 64, 64)).sum().backward()
    for conv in (bf.lateral2, bf.lateral3, bf.smooth, bf.layer3[0].conv1, bf.layer2[0].conv1):
        assert conv.weight.grad is not None
    assert bf.layer1[0].conv1.weight.grad is None
    assert bf.layer3[0].bn1.weight.grad is None  # BatchNorm frozen everywhere
