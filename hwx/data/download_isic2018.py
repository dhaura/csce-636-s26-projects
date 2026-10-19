#!/usr/bin/env python3
"""Download and extract the ISIC 2018 Task 3 (lesion diagnosis / HAM10000) dataset.

Uses only the Python standard library so it can run before the conda env exists.

Result layout (relative to this file's directory):
    isic2018/
        ISIC2018_Task3_Training_Input/        10015 .jpg (600x450)
        ISIC2018_Task3_Training_GroundTruth/  one-hot labels CSV
        ISIC2018_Task3_Training_LesionGroupings.csv   image -> lesion_id
        ISIC2018_Task3_Validation_Input/      193 .jpg  (official validation)
        ISIC2018_Task3_Validation_GroundTruth/
        ISIC2018_Task3_Test_Input/            1512 .jpg (official test)
        ISIC2018_Task3_Test_GroundTruth/

Usage:
    python data/download_isic2018.py [--keep-zips]

Data license: CC-BY-NC 4.0 (see LICENSE.txt / ATTRIBUTION.txt inside the image folders).
Source: https://challenge.isic-archive.com/data/#2018
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://isic-challenge-data.s3.amazonaws.com/2018"
ZIPS = [
    "ISIC2018_Task3_Training_Input.zip",
    "ISIC2018_Task3_Training_GroundTruth.zip",
    "ISIC2018_Task3_Validation_Input.zip",
    "ISIC2018_Task3_Validation_GroundTruth.zip",
    "ISIC2018_Task3_Test_Input.zip",
    "ISIC2018_Task3_Test_GroundTruth.zip",
]
CSVS = ["ISIC2018_Task3_Training_LesionGroupings.csv"]
EXPECTED_IMAGES = {
    "ISIC2018_Task3_Training_Input": 10015,
    "ISIC2018_Task3_Validation_Input": 193,
    "ISIC2018_Task3_Test_Input": 1512,
}

ROOT = Path(__file__).resolve().parent / "isic2018"
ZIP_DIR = Path(__file__).resolve().parent / "zips"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done, next_report = 0, 0.0
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"       {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)", flush=True)
                next_report += 0.1
    tmp.rename(dest)


def extract(zip_path: Path, out_dir: Path) -> None:
    target = out_dir / zip_path.stem
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {target.name} already extracted")
        return
    print(f"[unzip] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def verify() -> bool:
    ok = True
    for folder, n_expected in EXPECTED_IMAGES.items():
        n = len(list((ROOT / folder).glob("*.jpg")))
        status = "ok" if n == n_expected else "MISMATCH"
        ok &= n == n_expected
        print(f"[check] {folder}: {n} images (expected {n_expected}) {status}")
    for split in ("Training", "Validation", "Test"):
        csv = ROOT / f"ISIC2018_Task3_{split}_GroundTruth" / f"ISIC2018_Task3_{split}_GroundTruth.csv"
        ok &= csv.exists()
        print(f"[check] {csv.name}: {'ok' if csv.exists() else 'MISSING'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-zips", action="store_true", help="keep the downloaded .zip files after extraction")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for name in CSVS:
        download(f"{BASE_URL}/{name}", ROOT / name)
    for name in ZIPS:
        zip_path = ZIP_DIR / name
        if (ROOT / zip_path.stem).exists():
            print(f"[skip] {zip_path.stem} already extracted")
            continue
        download(f"{BASE_URL}/{name}", zip_path)
        extract(zip_path, ROOT)

    if not args.keep_zips:
        shutil.rmtree(ZIP_DIR, ignore_errors=True)

    if not verify():
        sys.exit("Dataset verification failed - re-run the script (completed files are skipped).")
    print(f"Done. Dataset ready at {ROOT}")


if __name__ == "__main__":
    main()
