"""
fetch_url.py — download a blog post / article and save its extracted text
into sample_docs/ as a .txt file, ready for build_index.py to pick up.

Usage:
    python fetch_url.py https://example.com/some-blog-post
"""

import os
import sys

from rag.ingest import extract_url_text, slugify

OUTPUT_FOLDER = "sample_docs"


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_url.py <url>")
        return

    url = sys.argv[1]
    print(f"Fetching {url} ...")

    try:
        title, text = extract_url_text(url)
    except ValueError as e:
        print(str(e))
        return

    filename = slugify(title) + ".txt"
    path = os.path.join(OUTPUT_FOLDER, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\nSource: {url}\n\n{text}")

    print(f"Saved {len(text.split())} words to {path}")
    print("Run 'python build_index.py' again to include it in your index.")


if __name__ == "__main__":
    main()
