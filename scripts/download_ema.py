#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EMA_DIR = Path("artifacts/ema/downloaded")

EMA_FILES = [
    "medicines.json",
    "post_authorisation.json",
    "referrals.json",
    "psusas.json",
    "dhpcs.json",
    "shortages.json",
]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ema_dir = EMA_DIR.resolve()
    ema_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BIOMCP_EMA_DIR"] = str(ema_dir)

    print(f"Using EMA directory: {ema_dir}")
    print("Downloading EMA data with BioMCP...")
    run(["biomcp", "ema", "sync"], env=env)

    print("\nChecking downloaded files:")
    missing = []
    for name in EMA_FILES:
        path = ema_dir / name
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  OK   {name} ({size_mb:.2f} MB)")
        else:
            print(f"  MISS {name}")
            missing.append(name)

    if missing:
        print("\nSome expected files are missing:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    print("\nEMA data downloaded successfully.")


if __name__ == "__main__":
    main()