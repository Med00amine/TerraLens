#!/usr/bin/env bash
# Model comparison used for the README results table (RTX 4060, ~1 h total).
set -e
for spec in resnet18:64 resnet18:128 resnet18:224 efficientnet_b0:128 mobilenet_v3_large:128 resnet50:128; do
  model=${spec%%:*}; size=${spec##*:}
  out=runs/$model-$size
  mkdir -p "$out"
  echo "=== $model @ ${size}px"
  python -m terralens.train --model "$model" --img-size "$size" --num-workers 8 --output-dir "$out" 2>&1 \
    | grep -v -e "Downloading" -e "%|" | tee "$out/train.log"
done
