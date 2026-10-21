"""Build the WattBot PDF corpus once, for staging in the shared bucket.

Reads the `url` column of the WattBot metadata.csv, downloads every PDF as
<id>.pdf, zips them into corpus.zip, then extracts every page's text with
pypdf into pages.jsonl (one JSON record per page: text, doc_id, title, url,
page_num, page_label, total_pages). Instructors run this once per term;
learners never download PDFs from publishers themselves (fifty notebooks
hitting the same publisher sites at once is slow and gets throttled), and
the Bedrock RAG episode loads pages.jsonl instead of spending minutes on
extraction.

    python scripts/build_wattbot_corpus.py --metadata data/wattbot/metadata.csv --out wattbot/corpus.zip

To add pages.jsonl to a corpus.zip that already exists, skip the downloads:

    python scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --from-zip wattbot/corpus.zip

Then upload corpus.zip, pages.jsonl, metadata.csv and train_QA.csv to
s3://<shared-bucket>/wattbot/ (see instructors/shared-data-bucket.md).
"""

import argparse
import json
import os
import tempfile
import time
import zipfile

import pandas as pd
import requests


def extract_pages(zip_path, md, pages_path):
    """Write one JSON record per PDF page to pages_path; the same records the episode's pdfs_to_page_docs builds."""
    import logging
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)   # pypdf warns about every embedded font it cannot fully parse; the text still extracts
    t0 = time.time()
    n = 0
    with tempfile.TemporaryDirectory() as tmp, open(pages_path, "w") as out:
        zipfile.ZipFile(zip_path).extractall(tmp)
        for _, row in md.iterrows():
            doc_id = str(row["id"]).strip()
            path = os.path.join(tmp, f"{doc_id}.pdf")
            if not os.path.exists(path):
                continue
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:  # noqa: BLE001
                    text = ""
                if not text:
                    text = "[[EMPTY PAGE TEXT \u2013 see original PDF for tables/figures]]"
                out.write(json.dumps({"text": text, "doc_id": doc_id, "title": str(row.get("title", "")).strip(),
                                      "url": str(row.get("url", "")).strip(), "page_num": i,
                                      "page_label": str(i + 1), "total_pages": len(reader.pages)}) + "\n")
                n += 1
    print(f"wrote {pages_path}: {n} pages in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="WattBot metadata.csv (has id and url columns)")
    ap.add_argument("--out", default="corpus.zip")
    ap.add_argument("--from-zip", help="an existing corpus.zip: skip the downloads and only write pages.jsonl")
    ap.add_argument("--pages", help="where to write the page records (default: pages.jsonl next to the zip)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retry-missing", action="store_true", help="keep the PDFs already in --out and download only the ids it lacks")
    args = ap.parse_args()

    md = pd.read_csv(args.metadata, encoding="utf-8", encoding_errors="replace")
    zip_path = args.from_zip or args.out
    pages_path = args.pages or os.path.join(os.path.dirname(os.path.abspath(zip_path)), "pages.jsonl")
    if args.from_zip:
        extract_pages(zip_path, md, pages_path)
        return
    failures = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mode = "a" if args.retry_missing and os.path.exists(args.out) else "w"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}   # some publishers refuse the default python-requests agent
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.out, mode, zipfile.ZIP_DEFLATED) as zf:
        have = {n[:-4] for n in zf.namelist() if n.endswith(".pdf")}
        for _, row in md.iterrows():
            doc_id = str(row["id"]).strip()
            if doc_id in have:
                continue
            url = row.get("url")
            if not isinstance(url, str) or not url.strip().startswith("http"):
                failures.append((doc_id, f"bad url: {url!r}"))
                continue
            url = url.strip().replace("arxiv.org/abs/", "arxiv.org/pdf/")   # the abstract page is HTML; the pdf link is the paper
            try:
                resp = requests.get(url, timeout=args.timeout, allow_redirects=True, headers=headers)
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
    extract_pages(args.out, md, pages_path)
    if failures:
        print(f"{len(failures)} failed; fix or accept before staging:")
        for d, e in failures:
            print(f"  {d}: {e}")


if __name__ == "__main__":
    main()
