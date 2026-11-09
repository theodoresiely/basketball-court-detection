# courtdet — basketball court detection in satellite imagery with a hand-written Faster R-CNN

A two-stage detector (ResNet-18 backbone, Region Proposal Network, RoI Align head) written piece by piece in
PyTorch to find basketball courts in aerial / satellite images. It is evaluated with a COCO-style mAP on whole
images, with bootstrap confidence intervals, and compared to a YOLOv8n baseline trained on the same pixels and
scored by the same evaluator.

## Results

All numbers come from runs whose output was read. Every modelling choice was made on the validation split; the test split was
evaluated **once**, at the end, with the two models fixed beforehand.

AP@0.5 and AP@[.5:.95] are computed like pycocotools, on whole images at native resolution, with 95 % bootstrap
intervals over images. "A" and "B" are the two image sources of the dataset (see *Data*).

### Test (151 images, 263 courts)

| | Faster R-CNN (ours) AP@0.5 | AP@[.5:.95] | YOLOv8n AP@0.5 | AP@[.5:.95] |
|---|---|---|---|---|
| all | 0.710 [0.584–0.860] | 0.448 [0.338–0.585] | 0.801 [0.686–0.892] | 0.585 [0.493–0.678] |
| A (70 images, 129 courts) | 0.909 [0.854–0.959] | 0.606 [0.548–0.671] | 0.905 [0.841–0.968] | 0.653 [0.587–0.724] |
| B (81 images, 134 courts) | 0.489 [0.265–0.752] | 0.294 [0.125–0.547] | 0.706 [0.430–0.877] | 0.531 [0.333–0.692] |

Paired bootstrap, YOLOv8n − ours: AP@[.5:.95] +0.137 [+0.044 ; +0.220].

### Validation (303 images, 317 courts)

| | AP@0.5 | AP@[.5:.95] |
|---|---|---|
| reference model | 0.884 [0.840–0.925] | 0.620 [0.578–0.662] |
| YOLOv8n | 0.904 [0.860–0.940] | 0.669 [0.627–0.710] |

### What the test showed: small courts

On source A the test agrees with validation. On source B the model drops sharply, and the reason is the size of
the courts, not overfitting to validation:

| B courts found (score ≥ 0.5, IoU ≥ 0.5) | < 32 px | 32–64 px | ≥ 64 px |
|---|---|---|---|
| reference model, valid | 0/10 | 4/7 | 71/74 |
| YOLOv8n, valid | 0/10 | 5/7 | 70/74 |
| reference model, test | 5/43 | 28/42 | 44/49 |
| YOLOv8n, test | 14/43 | 35/42 | 43/49 |

The median court side on B is 88 px in validation and 41 px in test. Small courts were almost absent from
validation, so every choice was blind to them. On large courts both models are equivalent and validation predicted
the test well. Small courts are the main weakness of the model; fixing them would need a new validation split that
contains them, and the current test numbers could no longer serve as an unbiased estimate afterwards.

## How the model got there (validation, paired bootstrap)

Each step was kept only if a paired bootstrap on validation supported it :

1. **Corrected ground truth**: source A boxes that grouped several courts were redrawn by hand, with
   one box per court, in small review web pages (`scripts/review_server.py`, `edit_server.py`). The validation and
   test splits were corrected by hand, never from model outputs.
2. **Checkpoint selected on validation mAP**, not loss: near the optimum the two disagree.
3. **Backbone `layer3`** instead of `layer2`: on B, +0.065 and +0.088 AP@[.5:.95] over two seeds.
4. **A fair YOLO baseline**. Ultralytics resizes the long side of every image to `imgsz`, which would have
   shrunk the 2048 px scenes 4×. Instead, native-resolution crops drawn with a crop sampler are exported for
   training (`scripts/export_yolo.py`), and prediction runs on whole native images scored by an evaluator
   (`scripts/predict_yolo.py`). YOLOv8n led by +0.105 AP@[.5:.95].
5. **Why YOLO led**: retrained without its online augmentations, YOLOv8n lost 0.124 AP@[.5:.95] and fell to
   the model's level (−0.018 [−0.052 ; +0.020] vs ours). A good part of the lead came from augmentations.
6. **Online augmentations for the model** (`src/courtdet/data/augment.py`): the 8 symmetries of the square
   (exact box coordinates; aerial images have no "up"), plus brightness, contrast and saturation jitter and grey
   levels. After training to convergence (30 epochs): +0.056 / +0.058 AP@[.5:.95] against two seeds of the previous
   model, mostly on A; B unchanged. About half of the gap to YOLO closed.

Tried and not kept:

- a fused FPN-style `layer2`+`layer3` feature map (smaller gain on B than `layer3` alone: +0.049 vs +0.065, one seed);
- hard negative mining on B (extra training crops around the model's false alarms on scenes without courts): no AP
  gain established, and the drop in false alarms stayed within the seed-to-seed noise. The code was removed from the
  repository (it remains in the git history).

## Model

`src/courtdet/models/`, one file per component, each with its tests:

- `backbone.py`: ImageNet ResNet-18 truncated at `layer2` (stride 8), `layer3` (stride 16) or a fused map.
- `anchors.py`: 6 sizes (32–192 px, ratio ≤ 1.5 between neighbours) × 3 aspect ratios.
- `box_coder.py`, `matching.py`: box deltas, IoU matching and balanced sampling of anchors / RoIs.
- `rpn.py`: objectness (1 logit, BCE) and box regression, then decoding, clipping and NMS into proposals.
- `roi_head.py`: `roi_align` (aligned, 7×7), 2 fully connected layers, court / background score and one box regression.
- `faster_rcnn.py`: the full model, configured by a strict YAML file (`src/courtdet/config.py`).

Box convention everywhere: continuous (x1, y1, x2, y2), width = x2 − x1 (no +1), as in `torchvision.ops`.

## Data

Roboflow COCO export (v2), one class `court`, two very different sources:

- **A**: 256×256 tiles, usually one court each.
- **B**: large aerial scenes (around 2048 px), most of them without any court (true negatives), with courts of
  very different sizes.

Images are used at **native resolution**. Training draws random 512×512 crops; a crop that would cut a court is
re-drawn, since a court must be either absent or at least 85 % visible (`src/courtdet/data/crops.py`). Evaluation
runs on whole images in one pass.

The data and the human review files (`data/review/`) are not part of the repository.

## Usage

```bash
pip install -e ".[dev]"      # package + pytest (and ".[yolo]" for the baseline, AGPL-3.0)
pytest -q                    # 189 tests, CPU
python scripts/train.py --config configs/augment_layer3.yaml --out runs/augment_layer3_e30_lr3e-4
python scripts/evaluate.py --checkpoint runs/augment_layer3_e30_lr3e-4/best.pt --split valid \
    --corrections data/review/corrections.json --decisions data/review/decisions.json
python scripts/predict.py --checkpoint runs/augment_layer3_e30_lr3e-4/best.pt --images <files or folder>  # draws the boxes
python scripts/export_yolo.py --config configs/corrected.yaml --out data/yolo/v2_corrected --views 4
python scripts/train_yolo.py --data data/yolo/v2_corrected/data.yaml --name v8n_e20 --epochs 20
python scripts/predict_yolo.py --weights runs/yolo/v8n_e20/weights/best.pt --compare <our eval .pt>
```

Trained on a MacBook Air (Apple M5, MPS): about 74 minutes for the reference model (30 epochs) and about
62 minutes for YOLOv8n (20 epochs).

## Limitations

- **Small courts** (see above): the smallest anchor is 32 px and the `layer3` map has stride 16. These are likely
  causes, not tested.
- **False alarms on B**: other sports grounds and roofs. On B images without courts, at score ≥ 0.5, the
  reference model gives 96 detections and YOLOv8n 4.
- **Small evaluation sets**: 91 B courts in validation, so intervals on B are about ±0.05 to ±0.1; differences
  below that cannot be established.
- **One seed per configuration** for most runs. The measured run-to-run noise is up to 0.03 AP@[.5:.95] on B,
  and MPS training is not reproducible at a fixed seed.
- Augmentations were not tuned, and scale augmentation was not tried.

## Repository

```
src/courtdet/   data/ (crops, dataset, augmentations, review and corrections), models/, engine/ (train, evaluate, error analysis)
scripts/        training, evaluation, prediction, YOLO export / train / predict, review web pages
configs/        YAML configs (default.yaml = every field with its default value)
tests/          pytest suite
```
