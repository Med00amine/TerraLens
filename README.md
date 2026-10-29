# TerraLens

Satellite land-cover classification and mapping with deep learning.

> 🚧 Work in progress. This repository started as my collection of PyTorch exercise notebooks (2025) and is being rebuilt into a complete land-cover analysis tool.

## Goal

Classify Sentinel-2 satellite imagery into 10 land-cover classes (forest, crops, water, urban areas, and more) and turn large scenes into land-cover maps, with explanations of what the model looked at.

## Roadmap

- [ ] Data pipeline (EuroSAT, stratified splits, augmentation)
- [ ] Transfer-learning model zoo and training loop
- [ ] Evaluation report and model comparison
- [ ] Grad-CAM explainability
- [ ] Large-scene land-cover mapping
- [ ] REST API (FastAPI)
- [ ] Web interface
- [ ] Docker image and model release

## License

[MIT](LICENSE)
