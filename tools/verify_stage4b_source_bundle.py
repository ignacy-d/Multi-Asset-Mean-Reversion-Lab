#!/usr/bin/env python3
"""Verify a retrieved Stage 4B bundle before Stage 4C ingestion."""

import argparse
import json
from pathlib import Path

from mr_lab.stage4b_source_bundle import authenticate_source_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument(
        "--approved-source-registry",
        type=Path,
        default=Path("configs/stage4b-approved-source-registry.json"),
    )
    args = parser.parse_args()
    paths, audit = authenticate_source_bundle(
        args.bundle_dir, args.approved_source_registry
    )
    print(
        json.dumps(
            {
                "authentication": audit["source_authentication"],
                "bundle_id": audit["source_bundle_id"],
                "instrument": audit["instrument"],
                "ordered_raw_files": [str(path) for path in paths],
                "source_bundle_manifest_sha256": audit[
                    "source_bundle_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
