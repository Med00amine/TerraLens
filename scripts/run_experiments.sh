#!/usr/bin/env bash
# Experiments behind the README results (RTX 4060; run one job at a time).
set -e

# 1. Backbone / resolution comparison: same recipe, no JPEG augmentation.
for spec in resnet18:64 resnet18:128 resnet18:224 efficientnet_b0:128 mobilenet_v3_large:128 resnet50:128; do
  model=${spec%%:*}; size=${spec##*:}
  terralens train --model "$model" --img-size "$size" --no-jpeg-aug --num-workers 4 \
    --output-dir "runs/$model-$size"
done

# 2. Deployed model: ResNet18 @ 128 px with JPEG augmentation (quality 50-95).
terralens train --model resnet18 --img-size 128 --num-workers 4 --output-dir runs/resnet18-128-jpeg

# 3. Augmented candidates that were not selected.
terralens train --model resnet18 --img-size 224 --jpeg-min-quality 75 --num-workers 4 \
  --output-dir runs/cand-resnet18-224-jpeg75
terralens train --model mobilenet_v3_large --img-size 128 --jpeg-min-quality 75 --num-workers 4 \
  --output-dir runs/abl-mobilenet-jpeg75

# 4. Reports.
terralens evaluate --compare runs/{resnet18-64,resnet18-128,resnet18-224,resnet50-128,efficientnet_b0-128,mobilenet_v3_large-128}/model.pt --out docs/results
for scene in lake_constance mantua; do
  python scripts/compare_worldcover.py --scene "$scene" --model runs/resnet18-128-jpeg/model.pt \
    --figure "docs/images/worldcover_$scene.png"
done
python scripts/robustness.py runs/resnet18-128/model.pt runs/resnet18-128-jpeg/model.pt
