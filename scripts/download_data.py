#!/usr/bin/env python
"""Downloads the official UTKFace dataset into data/raw.
Note: The data will be downloaded as compressed .tar.gz files. 

Quick guide:
1. The script maps each archive name to its Google Drive file ID.
2. It creates `data/raw/utkface/` if the directory does not exist.
3. It requests each file, handling Google Drive's large-file confirm page.
4. It streams bytes into a temporary `.part` file, then renames it when done.
5. It skips files already on disk unless `--force` is passed.

Below are a couple of examples. The one you want to use is probably the first one.

Examples:
    python scripts/download_data.py
    python scripts/download_data.py --files part3.tar.gz
"""

from __future__ import annotations

import argparse
from email.message import Message
import html
import re
import sys
from typing import Protocol, Self
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "utkface"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "Mozilla/5.0"
DOWNLOADS = {
    "part1.tar.gz": {
        "file_id": "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW",
        "size_mb": 833.4,
    },
    "part2.tar.gz": {
        "file_id": "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b",
        "size_mb": 437.3,
    },
    "part3.tar.gz": {
        "file_id": "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b",
        "size_mb": 54.3,
    },
}


class ResponseLike(Protocol):
    headers: Message

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UTKFace archives into data/raw/utkface."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded archives (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=sorted(DOWNLOADS),
        default=sorted(DOWNLOADS),
        help="Optional subset of archive names to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser.parse_args()


def build_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def open_drive_stream(file_id: str) -> ResponseLike:
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = urllib.request.urlopen(build_request(initial_url))

    if response.headers.get("Content-Disposition"):
        return response

    page = response.read().decode("utf-8", errors="ignore")
    response.close()

    action_match = re.search(r'<form id="download-form" action="([^"]+)"', page)
    hidden_fields = dict(
        re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page)
    )
    if not action_match or not hidden_fields:
        raise RuntimeError(f"Google Drive did not return a downloadable file for {file_id}.")

    download_url = (
        html.unescape(action_match.group(1))
        + "?"
        + urllib.parse.urlencode(hidden_fields)
    )
    return urllib.request.urlopen(build_request(download_url))


def stream_download(file_name: str, file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with open_drive_stream(file_id) as response, temp_path.open("wb") as handle:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = (downloaded / total_bytes) * 100
                    sys.stdout.write(
                        f"\rDownloading {file_name}: {downloaded / 1024 / 1024:.1f} / "
                        f"{total_bytes / 1024 / 1024:.1f} MB ({percent:.1f}%)"
                    )
                else:
                    sys.stdout.write(
                        f"\rDownloading {file_name}: {downloaded / 1024 / 1024:.1f} MB"
                    )
                sys.stdout.flush()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(destination)
    sys.stdout.write("\n")
    print(f"Saved {file_name} to {destination}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    print(f"Downloading UTKFace archives into {output_dir}")

    for file_name in args.files:
        destination = output_dir / file_name
        if destination.exists() and not args.force:
            print(f"Skipping {file_name}; already exists at {destination}")
            continue

        metadata = DOWNLOADS[file_name]
        print(
            f"Starting {file_name} ({metadata['size_mb']} MB) from the official UTKFace source..."
        )
        stream_download(file_name, metadata["file_id"], destination)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
