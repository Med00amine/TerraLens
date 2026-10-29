import pytest

from terralens.download import download_model, sha256sum


@pytest.fixture
def remote(tmp_path):
    """A fake release asset served through a file:// URL."""
    src = tmp_path / "remote" / "model.pt"
    src.parent.mkdir()
    src.write_bytes(b"weights" * 1000)
    return src, src.as_uri()


def test_download_verifies_checksum(remote, tmp_path):
    src, url = remote
    dest = download_model(url, tmp_path / "models" / "m.pt", sha256=sha256sum(src))
    assert dest.read_bytes() == src.read_bytes()


def test_checksum_mismatch_leaves_nothing_behind(remote, tmp_path):
    _, url = remote
    dest = tmp_path / "models" / "m.pt"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_model(url, dest, sha256="0" * 64)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part"))


def test_existing_valid_file_is_not_downloaded_again(remote, tmp_path):
    src, url = remote
    dest = tmp_path / "m.pt"
    dest.write_bytes(src.read_bytes())
    mtime = dest.stat().st_mtime_ns
    download_model("file:///does/not/exist", dest, sha256=sha256sum(src))
    assert dest.stat().st_mtime_ns == mtime


def test_stale_file_is_replaced(remote, tmp_path):
    src, url = remote
    dest = tmp_path / "m.pt"
    dest.write_bytes(b"old")
    download_model(url, dest, sha256=sha256sum(src))
    assert dest.read_bytes() == src.read_bytes()


def test_default_destination_follows_env(remote, tmp_path, monkeypatch):
    _, url = remote
    monkeypatch.setenv("TERRALENS_MODEL", str(tmp_path / "env" / "model.pt"))
    assert download_model(url, sha256=None) == tmp_path / "env" / "model.pt"
