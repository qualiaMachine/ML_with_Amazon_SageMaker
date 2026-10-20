"""Build the WattBot PDF corpus once, for staging in the shared bucket.

Reads the `url` column of the WattBot metadata.csv, downloads every PDF as
<id>.pdf, and zips them into corpus.zip. Instructors run this once per term;
learners never download PDFs from publishers themselves (fifty notebooks
hitting the same publisher sites at once is slow and gets throttled).

    python scripts/build_wattbot_corpus.py --metadata data/wattbot/metadata.csv --out wattbot/corpus.zip

Then upload corpus.zip, metadata.csv and train_QA.csv to
s3://<shared-bucket>/wattbot/ (see instructors/shared-data-bucket.md).
"""

import argparse
import os
import tempfile
import zipfile

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="WattBot metadata.csv (has id and url columns)")
    ap.add_argument("--out", default="corpus.zip")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    md = pd.read_csv(args.metadata, encoding="utf-8", encoding_errors="replace")
    failures = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in md.iterrows():
            doc_id = str(row["id"]).strip()
            url = row.get("url")
            if not isinstance(url, str) or not url.strip().startswith("http"):
                failures.append((doc_id, f"bad url: {url!r}"))
                continue
            try:
                resp = requests.get(url.strip(), timeout=args.timeout, allow_redirects=True)
                resp.raise_for_status()
                path = os.path.join(tmp, f"{doc_id}.pdf")
                with open(path, "wb") as f:
                    f.write(resp.content)
                zf.write(path, arcname=f"{doc_id}.pdf")
                print(f"  ok   {doc_id}  ({len(resp.content) / 1e6:.1f} MB)")
            except Exception as e:  # noqa: BLE001
                failures.append((doc_id, str(e)))
                print(f"  FAIL {doc_id}: {e}")
    print(f"\nwrote {args.out} with {len(md) - len(failures)} PDFs")
    if failures:
        print(f"{len(failures)} failed; fix or accept before staging:")
        for d, e in failures:
            print(f"  {d}: {e}")


if __name__ == "__main__":
    main()
