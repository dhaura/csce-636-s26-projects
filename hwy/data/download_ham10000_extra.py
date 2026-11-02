#!/usr/bin/env python
"""Download the HAM10000 extras used in Homework Y into data/ham10000_extra/ (about 12 MB).

From the HAM10000 dataset on Harvard Dataverse (doi:10.7910/DVN/DBW86T; CC BY-NC 4.0, non-commercial use):
  HAM10000_metadata.tab                     age, sex, body site of the 10,015 training images
  ISIC2018_Task3_Test_GroundTruth.tab       the same fields (+ diagnosis) for the 1,512 official test images
  HAM10000_segmentations_lesion_tschandl    lesion masks of the training images (unzipped into masks/)

Despite the .tab extension the tables are comma-separated. The images themselves come from data/isic2018/
(shared with Homework X via setup_links.sh).

    python data/download_ham10000_extra.py
"""
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "ham10000_extra"
FILES = {  # Dataverse datafile id -> local name
    4338392: "HAM10000_metadata.tab",
    6924466: "ISIC2018_Task3_Test_GroundTruth.tab",
    3838943: "HAM10000_segmentations_lesion_tschandl.zip",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fid, name in FILES.items():
        dst = OUT / name
        if not dst.exists():
            url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original"
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, dst)
    masks = OUT / "masks"
    if not (masks / "HAM10000_segmentations_lesion_tschandl").is_dir():
        with zipfile.ZipFile(OUT / FILES[3838943]) as z:
            z.extractall(masks)
    n = len(list((masks / "HAM10000_segmentations_lesion_tschandl").glob("*_segmentation.png")))
    print(f"done: {n} lesion masks (expected 10015) in {masks}")


if __name__ == "__main__":
    main()
