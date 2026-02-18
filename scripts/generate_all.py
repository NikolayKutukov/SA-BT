#!/usr/bin/env python
"""Generate all 5 synthetic datasets and save to data/."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate import (
    generate_setup_1,
    generate_setup_2,
    generate_setup_3,
    generate_setup_4,
    generate_setup_5,
)

GENERATORS = {
    "setup_1": generate_setup_1,
    "setup_2": generate_setup_2,
    "setup_3": generate_setup_3,
    "setup_4": generate_setup_4,
    "setup_5": generate_setup_5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic survival datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory (default: data/)")
    parser.add_argument(
        "--setups",
        nargs="*",
        default=list(GENERATORS.keys()),
        choices=list(GENERATORS.keys()),
        help="Which setups to generate (default: all)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(args.setups):
        print(f"\n{'='*60}")
        print(f"Generating {name} …")
        t0 = time.perf_counter()

        data = GENERATORS[name](seed=args.seed + i)

        elapsed = time.perf_counter() - t0
        print(data.summary())
        print(f"  generation time: {elapsed:.2f}s")

        # Save as CSV
        csv_path = outdir / f"{name}.csv"
        data.to_dataframe().to_csv(csv_path, index=False)
        print(f"  saved to {csv_path}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
