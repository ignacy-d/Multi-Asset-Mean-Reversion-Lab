#!/usr/bin/env python3
"""Build (but never approve) a deterministic Stage 4B source-bundle candidate."""

import argparse
import json
from pathlib import Path

from mr_lab.stage4b_source_bundle import build_candidate_source_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--combined-audit-filename", default="execution-audit.json")
    args = parser.parse_args()
    provenance = json.loads(args.source_provenance.read_text())
    path, digest = build_candidate_source_bundle(
        args.bundle_dir,
        provenance,
        combined_audit_filename=args.combined_audit_filename,
    )
    print(
        json.dumps({"candidate_manifest": str(path), "sha256": digest}, sort_keys=True)
    )
    print("UNAPPROVED CANDIDATE: independent review and registry commit required")


if __name__ == "__main__":
    main()
