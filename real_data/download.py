"""Download and verify a fixed subset of Sleep-EDF Expanded v1.0.0.

Only public filenames and official SHA-256 digests are shipped in this
repository. The EDF files remain subject to PhysioNet's dataset license.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
RESOURCE_ROOT = Path(__file__).resolve().parent


def selected_files(nights: str) -> list[str]:
    """Select both nights or the first available night of each subject."""
    if nights not in {"all", "first"}:
        raise ValueError("nights must be 'all' or 'first'.")
    names = (RESOURCE_ROOT / "download_manifest.txt").read_text().splitlines()
    if nights == "all":
        return names
    first_nights = {}
    for name in sorted(names):
        first_nights.setdefault(name[:5], name[:6])
    return [name for name in names if name[:6] == first_nights[name[:5]]]


def read_checksums() -> dict[str, str]:
    checksums = {}
    for line in (RESOURCE_ROOT / "SHA256SUMS.txt").read_text().splitlines():
        digest, filename = line.split()
        checksums[Path(filename).name] = digest
    return checksums


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(filename: str, target: Path, expected: str) -> None:
    """Resume a partial file and atomically publish it after checksum validation."""
    if target.exists() and sha256(target) == expected:
        print(f"Already verified: {filename}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists() and sha256(partial) == expected:
        partial.replace(target)
        print(f"Verified completed partial: {filename}")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(f"{BASE_URL}/{filename}", headers={"User-Agent": "FENet-research"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urlopen(request, timeout=120) as response:
        # A server may ignore Range; restart instead of appending a second copy.
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if sha256(partial) != expected:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {filename}; rerun the download.")
    partial.replace(target)
    print(f"Downloaded and verified: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sleep-edf"))
    parser.add_argument("--nights", choices=("first", "all"), default="all")
    parser.add_argument("--list-only", action="store_true", help="Print filenames without downloading.")
    args = parser.parse_args()
    names = selected_files(args.nights)
    if args.list_only:
        print("\n".join(names))
        return
    target_dir = args.data_root.expanduser().resolve() / "raw" / "sleep-cassette"
    target_dir.mkdir(parents=True, exist_ok=True)
    checksums = read_checksums()
    for filename in names:
        download_file(filename, target_dir / filename, checksums[filename])


if __name__ == "__main__":
    main()
