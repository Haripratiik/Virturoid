"""Generate the data-engine demo artifact (YC demo plan item 7): the MimicGen factory turns ONE scripted grasp
into N VERIFIED demos via randomization + rejection sampling -- the answer to "where is your robot-action data?".
Writes reports/grasp_dataset_summary.json into a package so the Reports card can show the augmentation headline.

Usage:  python scripts/demo_data_factory.py [--package build/ui_verify/arm_sort] [--n 48]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VIRTUROID_LLM_BACKEND", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virturoid.services.data_factory import generate_grasp_demos  # noqa: E402
from virturoid.services.morphology_composer import compose_robot  # noqa: E402


def main(package: str = "build/ui_verify/arm_sort", n: int = 48) -> int:
    gene = compose_robot("grasp and lift a box on a table", llm=None)
    ds = generate_grasp_demos(gene, n=n)
    summary = {k: v for k, v in ds.items() if k != "demos"}
    summary["headline"] = (f"{ds['augmentation_x']} verified demos from 1 scripted grasp "
                           f"({int(ds['yield'] * 100)}% yield) - the in-sim data engine")
    out = Path(package) / "reports" / "grasp_dataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary["headline"], "->", out)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the MimicGen data-factory demo artifact.")
    ap.add_argument("--package", default="build/ui_verify/arm_sort")
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    raise SystemExit(main(args.package, args.n))
