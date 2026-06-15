from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


IMAGENET_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
)
OFFICIAL_GDRIVE_ID = "1S7Upn_8dWHNN5R3woazpocFU6J8hvCIe"
OFFICIAL_GDRIVE_URL = "https://drive.google.com/file/d/1S7Upn_8dWHNN5R3woazpocFU6J8hvCIe/view?usp=share_link"


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def download_url(url: str, output: Path, force: bool = False) -> None:
    if _nonempty(output) and not force:
        print(f"exists, skip: {output.resolve()}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        if not _nonempty(tmp):
            raise RuntimeError("downloaded file is empty")
        tmp.replace(output)
        print(f"saved: {output.resolve()}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"manual URL: {url}", file=sys.stderr)
        raise


def download_gdrive(file_id: str, output: Path, force: bool = False) -> None:
    if _nonempty(output) and not force:
        print(f"exists, skip: {output.resolve()}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gdown", file_id, "-O", str(output)]
    try:
        subprocess.run(cmd, check=True)
        if not _nonempty(output):
            if output.exists():
                output.unlink()
            raise RuntimeError("gdown produced an empty file")
        print(f"saved: {output.resolve()}")
    except Exception as exc:
        if output.exists() and output.stat().st_size == 0:
            output.unlink()
        print(f"Google Drive download failed: {exc}", file=sys.stderr)
        print("Install gdown with: python -m pip install gdown", file=sys.stderr)
        print(f"manual URL: {OFFICIAL_GDRIVE_URL}", file=sys.stderr)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Download PMT reproduction weights")
    parser.add_argument("--output-dir", default="pretrained")
    parser.add_argument("--imagenet", action="store_true", help="download ImageNet ViT-B/16 weight")
    parser.add_argument("--official", action="store_true", help="download official PMT SYSU weight")
    parser.add_argument("--all", action="store_true", help="download both weights")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    want_imagenet = args.all or args.imagenet or not (args.imagenet or args.official)
    want_official = args.all or args.official
    if want_imagenet:
        download_url(IMAGENET_URL, output_dir / "jx_vit_base_p16_224-80ecf9dd.pth", force=args.force)
    if want_official:
        download_gdrive(OFFICIAL_GDRIVE_ID, output_dir / "pmt_sysu_vit_official.pth", force=args.force)


if __name__ == "__main__":
    main()

