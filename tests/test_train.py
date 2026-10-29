import json

import torch

from terralens.train import TrainConfig, parse_args, train


def test_parse_args_defaults_and_overrides():
    cfg = parse_args(["--model", "efficientnet_b0", "--img-size", "96", "--lr", "3e-4", "--no-amp"])
    assert cfg.model == "efficientnet_b0"
    assert cfg.img_size == 96
    assert cfg.lr == 3e-4
    assert cfg.amp is False
    assert cfg.output_dir == "runs/efficientnet_b0-96"
    assert cfg.epochs == TrainConfig().epochs


def test_train_end_to_end_on_tiny_dataset(image_folder, tmp_path):
    out = tmp_path / "run"
    cfg = TrainConfig(
        dataset="folder",
        data_root=str(image_folder),
        download=False,
        model="resnet18",
        pretrained=False,
        img_size=32,
        epochs=2,
        batch_size=8,
        num_workers=0,
        amp=False,
        output_dir=str(out),
    )
    summary = train(cfg)

    assert {p.name for p in out.iterdir()} == {"model.pt", "history.json", "metrics.json"}
    assert len(json.loads((out / "history.json").read_text())) == summary["epochs_run"] == 2
    assert 0.0 <= summary["test_acc"] <= 1.0

    ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    assert ckpt["architecture"] == "resnet18"
    assert ckpt["class_names"] == ["Forest", "River", "SeaLake"]
    assert ckpt["img_size"] == 32
    assert ckpt["metrics"]["test_acc"] == summary["test_acc"]
    assert ckpt["config"]["seed"] == 42
