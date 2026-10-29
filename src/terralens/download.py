"""Download the released model weights.

terralens download-model            # -> models/terralens.pt (or $TERRALENS_MODEL)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from terralens.predictor import default_model_path

RELEASE_URL = "https://github.com/Med00amine/TerraLens/releases/download/v1.0.0/terralens-v1.0.0.pt"
RELEASE_SHA256 = None  # set when the v1.0.0 weights are published


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    url: str = RELEASE_URL,
    dest: str | Path | None = None,
    sha256: str | None = RELEASE_SHA256,
    force: bool = False,
) -> Path:
    """Fetch `url` to `dest`, verifying the checksum; skips files that are already valid.

    The download goes to a temporary file in the destination folder and is only
    moved into place once complete and verified, so an interrupted download never
    leaves a corrupt checkpoint behind.
    """
    dest = Path(dest) if dest is not None else default_model_path()
    if dest.is_file() and not force and (sha256 is None or sha256sum(dest) == sha256):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
    tmp_path = Path(tmp.name)
    if sha256 is not None and sha256sum(tmp_path) != sha256:
        tmp_path.unlink()
        raise ValueError(f"Checksum mismatch for {url}")
    os.replace(tmp_path, dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the released TerraLens model.")
    parser.add_argument("--url", default=RELEASE_URL)
    parser.add_argument("--out", type=Path, default=None, help="default: $TERRALENS_MODEL")
    parser.add_argument("--sha256", default=RELEASE_SHA256, help="expected checksum")
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args(argv)
    path = download_model(args.url, args.out, args.sha256, args.force)
    print(f"Model ready at {path} ({path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
