# Model card: TerraLens v1.0.0

## Model details

| | |
|---|---|
| Task | Land-cover classification of Sentinel-2 RGB tiles (64 × 64 px = 640 m × 640 m), and scene mapping by tiling |
| Architecture | ResNet18 (ImageNet-pretrained, torchvision), 10-way linear head |
| Input | RGB, resized to 128 px, ImageNet normalisation |
| Output | Calibrated probabilities over 10 classes (temperature scaling, T = 0.575) |
| Training | 12 epochs, AdamW + OneCycle, label smoothing 0.1, mixed precision, early stopping on validation accuracy; flips, 90° rotations, colour jitter, random JPEG re-compression (quality 50–95) |
| Weights | [GitHub release v1.0.0](https://github.com/Med00amine/TerraLens/releases/tag/v1.0.0) (`terralens download-model`) |
| Author | Mohamed Amine Bouzoffara |
| License | MIT (code and weights) |

## Intended use

- Quick land-cover inventories and area estimates over scenes of a few hundred km².
- Exploratory change analysis (compare maps of the same area at two dates).
- Teaching and demonstrating remote-sensing and explainable-AI concepts.

Predictions are **estimates**. Use them to explore and prioritise, not as an authoritative map for legal, financial or safety-critical decisions.

## Out-of-scope use

- Imagery that is not ~10 m/pixel optical RGB (e.g. drone photos, SAR, very-high-resolution aerial images) without adjusting `tile_px` and re-validating.
- Regions and land-cover types far from the European training data (tropical forest, deserts, snow and ice are not among the classes).
- Parcel-level or legal boundary decisions: one label describes a whole 640 m tile.

## Training data

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019): 27,000 Sentinel-2 L1C RGB tiles from 34 European countries, 10 classes, 2,000–3,000 images per class. Stratified split 70 / 15 / 15 (seed 42): 18,900 train, 4,050 validation, 4,050 test.

## Evaluation

| Metric | Value |
|---|---|
| EuroSAT test accuracy (4,050 held-out tiles) | 98.42% |
| Macro F1 | 0.983 |
| Expected calibration error (test) | 0.005 (0.086 before temperature scaling) |
| Accuracy on test tiles re-compressed as JPEG q75 / q50 | 97.98% / 96.99% |
| Agreement with ESA WorldCover on real L2A scenes, homogeneous tiles | 83.0% pooled over 2 scenes (Lake Constance 90.4%, Mantua 75.0%) |
| Agreement with ESA WorldCover on real L2A scenes, all tiles | 57.7% pooled (Lake Constance 58.6%, Mantua 56.8%) |

The model was selected among 9 candidates (6 backbones, 3 augmentation variants) by the best real-world agreement among those keeping ≥ 98.3% EuroSAT accuracy and ≥ 95% accuracy at JPEG quality 50. ResNet50 had the best EuroSAT score (99.01%) but lower real-world agreement (77.5%). Details, per-class results and figures are in the [README](README.md#results); raw numbers are in [`docs/results/`](docs/results).

## Known limitations and failure modes

- **Mixed tiles.** A 640 m tile containing a village in farmland gets one label (often *Residential*), while pixel-level maps such as ESA WorldCover call most of it vegetation. Agreement on homogeneous tiles is far higher than on all tiles for this reason.
- **Hail-net orchards** (common around Lake Constance) look like grey industrial roofs at 10 m and can be labelled *Industrial*.
- **Single date.** Young crop fields, pastures and orchards look alike in one image; WorldCover separates them with a year of observations. Vegetation classes are also the most confused on EuroSAT (F1 0.97 for pasture, permanent crop and herbaceous vegetation).
- **Algae-covered water** (e.g. the lakes around Mantua) can be labelled as vegetation.
- **Processing level.** EuroSAT is top-of-atmosphere (L1C) data; most users download atmospherically corrected (L2A) imagery. The model transfers well to unambiguous L2A tiles, but an augmentation that simulated the L1C→L2A colour shift made results worse and is disabled.
- **Compression.** Thanks to JPEG augmentation accuracy stays at 97% even at JPEG quality 50, but lossless input (PNG/GeoTIFF) remains the safest choice; the web UI says so when a JPEG is uploaded.

## Ethical considerations

The model works on public, coarse-resolution (10 m) imagery in which individual people, vehicles and buildings are not identifiable. The main risk is **over-trust**: a confident-looking map can hide errors. TerraLens mitigates this with calibrated probabilities, a confidence layer, an "uncertain area" statistic and Grad-CAM explanations, and states its limitations in the interface.
