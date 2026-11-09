import copy

import torch
from torch import nn
from torchvision.ops import box_iou

from courtdet.models.faster_rcnn import AnchorConfig, FasterRCNN, FasterRCNNConfig, RoIConfig

KEYS = {"rpn_cls", "rpn_reg", "roi_cls", "roi_reg", "total"}


def build(**kwargs):
    torch.manual_seed(0)
    return FasterRCNN(FasterRCNNConfig(pretrained_backbone=False, **kwargs))


def batch(size=128):
    torch.manual_seed(1)
    images = torch.randn(2, 3, size, size)
    gts = [torch.tensor([[20.0, 30.0, 70.0, 90.0]]), torch.zeros(0, 4)]  # one court, one empty image
    return images, gts


def test_forward_returns_finite_losses_and_total():
    model = build()
    losses = model(*batch())
    assert set(losses) == KEYS
    assert all(torch.isfinite(v) for v in losses.values())
    expected = losses["rpn_cls"] + losses["rpn_reg"] + losses["roi_cls"] + losses["roi_reg"]
    assert torch.allclose(losses["total"], expected)  # default lambdas = 1


def test_reg_weights_are_applied_to_the_total():
    model = build()
    model.config.rpn.reg_weight, model.config.roi.reg_weight = 10.0, 3.0
    torch.manual_seed(2)  # same samplings in both calls
    losses = model(*batch())
    expected = losses["rpn_cls"] + 10 * losses["rpn_reg"] + losses["roi_cls"] + 3 * losses["roi_reg"]
    assert torch.allclose(losses["total"], expected)


def test_every_trainable_parameter_gets_a_gradient():
    # The notebook's P1 bug: the RoI head was missing from the optimizer and never trained.
    model = build()
    model(*batch())["total"].backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith("roi_head.") for n, _ in trainable)
    for name, p in trainable:
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert name.startswith("backbone."), name  # only the backbone has frozen parts


def test_optimizer_step_moves_the_roi_head_and_not_the_frozen_layers():
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = copy.deepcopy(model.state_dict())
    model(*batch())["total"].backward()
    opt.step()
    after = model.state_dict()
    for name in ("roi_head.fc6.weight", "roi_head.cls_score.weight", "rpn_head.conv.weight", "backbone.layer2.0.conv1.weight"):
        assert not torch.equal(before[name], after[name]), name
    for name in ("backbone.conv1.weight", "backbone.layer1.0.conv1.weight", "backbone.layer2.0.bn1.running_mean", "base_anchors"):
        assert torch.equal(before[name], after[name]), name


def test_batchnorm_stays_frozen_in_train_mode():
    model = build().train()
    assert model.training and model.roi_head.training
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def test_state_dict_round_trip():
    model = build().eval()
    images, _ = batch()
    clone = FasterRCNN(FasterRCNNConfig(pretrained_backbone=False)).eval()
    clone.load_state_dict(model.state_dict())
    for a, b in zip(model.detect(images), clone.detect(images)):
        assert torch.equal(a["boxes"], b["boxes"]) and torch.equal(a["scores"], b["scores"])


def test_detect_output_and_validation_loss_in_eval_mode():
    model = build().eval()
    images, gts = batch()
    detections = model.detect(images)
    assert len(detections) == 2
    for d in detections:
        assert set(d) == {"boxes", "scores"}
        assert len(d["boxes"]) <= 100
        assert (d["boxes"] >= 0).all() and (d["boxes"] <= 128).all()
        assert (d["scores"][:-1] >= d["scores"][1:]).all()
    with torch.no_grad():  # validation loss: the reason why forward always returns losses
        losses = model(images, gts)
    assert set(losses) == KEYS and torch.isfinite(losses["total"])


def test_config_drives_the_bricks():
    model = build()
    assert model.base_anchors.shape == (18, 4)  # 6 sizes x 3 ratios
    assert model.rpn_head.cls_logits.out_channels == 18
    small = build(anchors=AnchorConfig(sizes=(32.0, 64.0), aspect_ratios=(1.0,)), roi=RoIConfig(roi_output_size=5))
    assert small.rpn_head.cls_logits.out_channels == 2
    assert small.rpn_head.bbox_deltas.out_channels == 8
    assert small.roi_head.fc6.in_features == 128 * 5 * 5
    assert set(small(*batch())) == KEYS


def test_overfits_a_mini_batch():
    # Wiring test: if every brick is connected, the model can memorise 2 images.
    # Measured on 28/10 (CPU, this seed): total 1.99 -> 0.054 after 200 steps (~12 s),
    # 0.027 after 300 steps. The 10 % threshold leaves a ~x3.7 margin at 200 steps.
    torch.manual_seed(0)
    model = FasterRCNN(FasterRCNNConfig(pretrained_backbone=False))
    images = torch.randn(2, 3, 128, 128)
    gts = [torch.tensor([[20.0, 30.0, 70.0, 90.0]]), torch.tensor([[60.0, 10.0, 110.0, 50.0], [10.0, 80.0, 50.0, 120.0]])]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(200):
        total = model(images, gts)["total"]
        first = total.item() if first is None else first
        opt.zero_grad()
        total.backward()
        opt.step()
    assert total.item() < 0.1 * first

    model.eval()
    for det, gt in zip(model.detect(images), gts):
        iou = box_iou(gt, det["boxes"][: len(gt)])
        assert (iou.max(dim=1).values > 0.7).all()  # every court found among the top detections


def test_full_model_runs_with_layer3_and_fused_backbones():
    for features in ("layer3", "fused"):
        model = build(backbone_features=features, trainable_layers=("layer2", "layer3"))
        assert model.roi_head.spatial_scale == 1.0 / model.backbone.stride
        losses = model(*batch())
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        assert model.backbone.layer3[0].conv1.weight.grad is not None
