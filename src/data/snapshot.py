"""Create a DVC-managed source snapshot and verify its content hash."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
SNAPSHOT_PATH = ROOT / "data" / "versioned" / "dataset.csv"


def snapshot_dataset(input_path: Path, output_path: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")
    source_digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_digest.update(chunk)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    digest = hashlib.sha256()
    with output_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source_digest.hexdigest():
        raise ValueError("Raw dataset changed while its DVC snapshot was being created")
    return {
        "source": str(input_path),
        "snapshot": str(output_path),
        "sha256": digest.hexdigest(),
        "size_bytes": output_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()
    print(json.dumps(snapshot_dataset(args.input, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
