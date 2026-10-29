import pytest

from terralens import __version__, cli


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommand_help_is_forwarded(capsys):
    with pytest.raises(SystemExit):
        cli.main(["train", "--help"])
    assert "--img-size" in capsys.readouterr().out


def test_unknown_command_fails():
    with pytest.raises(SystemExit) as exc:
        cli.main(["fly"])
    assert exc.value.code == 2


def test_predict_command(predictor, trained_checkpoint, tmp_path, capsys):
    from PIL import Image

    img = tmp_path / "tile.png"
    Image.new("RGB", (64, 64), (30, 90, 40)).save(img)
    cli.main(
        ["predict", str(img), "--model", str(trained_checkpoint), "--explain-dir", str(tmp_path)]
    )
    assert str(img) in capsys.readouterr().out
    assert (tmp_path / "tile_gradcam.png").is_file()


def test_map_command(trained_checkpoint, tmp_path, capsys):
    from PIL import Image

    img = tmp_path / "scene.png"
    Image.new("RGB", (128, 128), (30, 90, 40)).save(img)
    cli.main(["map", str(img), "--model", str(trained_checkpoint), "--out-dir", str(tmp_path)])
    assert "4 tiles" in capsys.readouterr().out
    for suffix in ("map", "overlay", "confidence"):
        assert (tmp_path / f"scene_{suffix}.png").is_file()
    assert (tmp_path / "scene_stats.json").is_file()
