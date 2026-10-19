"""Package a Badger Scribe data download into the zip assets the lesson's
GitHub Release expects, and print the command that publishes them.

The lesson does not keep the competition images in git (1.95 GB of JPEGs
would make every clone slow and blow through Git LFS limits). Instead each
data version is a GitHub Release on the lesson repository with three zip
assets, each well under GitHub's 2 GiB per-asset limit:

    badger-scribe-v<N>-labels.zip        train.csv, test.csv, metadata.csv, metric.py,
                                         sample_submission.csv, *.md   (< 1 MB)
    badger-scribe-v<N>-images-train.zip  images/<page_id>.jpg for every page in train.csv
    badger-scribe-v<N>-images-test.zip   images/<page_id>.jpg for every page in test.csv

Usage (from a folder containing the Kaggle download, i.e. train.csv and images/):

    python build_badger_scribe_release.py --src ~/Downloads/badger-scribe --version 9

Then follow the printed `gh release create` command (needs the GitHub CLI
logged in with write access to the repo; pass --repo to publish elsewhere). Unzipping all three assets
into one directory recreates the original layout, which is what the
episode's training job and the shared S3 bucket expect.
"""

import argparse
import os
import sys
import zipfile

import pandas as pd

LABEL_FILES = [
    "train.csv", "test.csv", "metadata.csv", "metric.py", "sample_submission.csv",
    "transcription_conventions.md", "RESOURCES.md", "WRITEUP_TEMPLATE.md",
    "CONTRIBUTION_TEMPLATE.md", "contributing_transcriptions.md",
]
DEFAULT_REPO = "qualiaMachine/ML_with_Amazon_SageMaker"


def add_images(zf, src, page_ids, label):
    missing = []
    for pid in page_ids:
        path = os.path.join(src, "images", f"{pid}.jpg")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        zf.write(path, arcname=f"images/{pid}.jpg")
    print(f"  {label}: {len(page_ids) - len(missing)} images" + (f", MISSING {len(missing)}: {missing[:5]}" if missing else ""))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with the Kaggle download (train.csv, images/, ...)")
    ap.add_argument("--version", required=True, type=int, help="Kaggle dataset version number, e.g. 9")
    ap.add_argument("--out", default="release", help="output folder for the zips")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo that hosts the release (owner/name)")
    args = ap.parse_args()

    v = f"v{args.version}"
    os.makedirs(args.out, exist_ok=True)
    train = pd.read_csv(os.path.join(args.src, "train.csv"), keep_default_na=False)
    test = pd.read_csv(os.path.join(args.src, "test.csv"), keep_default_na=False)
    print(f"source: {args.src}  ->  {len(train)} train pages, {len(test)} test pages")

    # JPEGs do not compress; ZIP_STORED keeps packaging fast.
    labels_zip = os.path.join(args.out, f"badger-scribe-{v}-labels.zip")
    with zipfile.ZipFile(labels_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        n = 0
        for f in LABEL_FILES:
            p = os.path.join(args.src, f)
            if os.path.exists(p):
                zf.write(p, arcname=f)
                n += 1
            else:
                print(f"  (no {f} in source; skipped)")
        print(f"  labels: {n} files")

    missing = []
    for split, df in (("train", train), ("test", test)):
        path = os.path.join(args.out, f"badger-scribe-{v}-images-{split}.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            missing += add_images(zf, args.src, list(df["page_id"]), split)

    print("\nassets:")
    for f in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, f)) / 1e9
        flag = "  <-- over GitHub's 2 GiB asset limit, split it" if size > 2.0 else ""
        print(f"  {f:<42} {size:6.2f} GB{flag}")
    if missing:
        print(f"\nWARNING: {len(missing)} page ids in the CSVs have no image; fix the source before publishing.")
        sys.exit(1)

    tag = f"badger-scribe-data-{v}"
    print(f"""
publish (GitHub CLI, logged in with write access to {args.repo}):

  gh release create {tag} \\
      --repo {args.repo} \\
      --title "Badger Scribe data {v}" \\
      --notes-file release_notes_{v}.md \\
      {args.out}/badger-scribe-{v}-labels.zip \\
      {args.out}/badger-scribe-{v}-images-train.zip \\
      {args.out}/badger-scribe-{v}-images-test.zip

download URL pattern the lesson uses:
  https://github.com/{args.repo}/releases/download/{tag}/badger-scribe-{v}-<asset>.zip
""")


if __name__ == "__main__":
    main()
