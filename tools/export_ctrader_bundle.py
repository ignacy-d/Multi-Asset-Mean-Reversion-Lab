"""Deterministically export the explicit M5A cTrader production allowlist."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "ctrader" / "MRLabController"
PACKAGE_FILES = (
    "__init__.py",
    "signals/__init__.py",
    "signals/contracts.py",
    "portfolio/__init__.py",
    "portfolio/contracts.py",
    "portfolio/identity.py",
    "portfolio/kernel.py",
    "portfolio/opportunity.py",
    "portfolio/planning.py",
    "portfolio/registry.py",
    "risk/__init__.py",
    "risk/contracts.py",
    "risk/kernel.py",
    "risk/policy.py",
    "risk/sizing.py",
    "execution/__init__.py",
    "execution/contracts.py",
    "execution/fsm.py",
    "runtime/__init__.py",
    "runtime/checkpoint.py",
    "runtime/contracts.py",
    "runtime/controller.py",
    "runtime/persistence.py",
    "runtime/reconcile.py",
    "integrations/__init__.py",
    "integrations/ctrader/__init__.py",
    "integrations/ctrader/account.py",
    "integrations/ctrader/broker.py",
    "integrations/ctrader/contracts.py",
    "integrations/ctrader/host.py",
    "integrations/ctrader/market.py",
    "integrations/ctrader/storage.py",
    "integrations/ctrader/telemetry.py",
)
TEMPLATE_FILES = ("MR Lab Controller_main.py", "MRLabController.cs", "README.md")


def export(destination: Path, *, dry_run: bool = False) -> tuple[str, ...]:
    destination = destination.resolve()
    if destination in {Path(destination.anchor), ROOT, ROOT.parent}:
        raise ValueError("refusing dangerous export destination")
    if destination.exists() and not destination.is_dir():
        raise ValueError("export destination must be a directory")
    outputs = tuple(TEMPLATE_FILES) + tuple(f"python/mr_lab/{p}" for p in PACKAGE_FILES)
    if dry_run:
        return outputs
    destination.mkdir(parents=True, exist_ok=True)
    managed = destination / "python" / "mr_lab"
    if managed.exists():
        shutil.rmtree(managed)
    for relative in TEMPLATE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATE / relative, target)
    source_package = ROOT / "src" / "mr_lab"
    for relative in PACKAGE_FILES:
        target = managed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_package / relative, target)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in export(args.destination, dry_run=args.dry_run):
        print(path)


if __name__ == "__main__":
    main()
