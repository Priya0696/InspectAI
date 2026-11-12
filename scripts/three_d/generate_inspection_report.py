
import argparse
import json
from pathlib import Path

from inspectai.inspection.pipe import inspect_pipe


DATA_DIR = Path("data/processed/pointclouds")
REPORT_DIR = Path("reports/3d")

DEFAULT_REFERENCE = DATA_DIR / "reference_pipe.ply"
DEFAULT_SCAN = DATA_DIR / "registered_defective_pipe.ply"


def main():
    parser = argparse.ArgumentParser(
        description="InspectAI 3D pipe inspection"
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
    )

    parser.add_argument(
        "--scan",
        type=Path,
        default=DEFAULT_SCAN,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")

    report = inspect_pipe(
        reference_path=args.reference,
        scan_path=args.scan,
        tolerance_mm=args.tolerance,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR /
        f"inspection_{args.scan.stem}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    measurement = report["measurement"]
    quality = report["quality_control"]

    print("\n=== INSPECTAI 3D INSPECTION REPORT ===")

    print(f"Scan: {args.scan.name}")

    print(
        f"Nominal radius: "
        f"{measurement['nominal_radius_mm']:.3f} mm"
    )

    print(
        f"Maximum dent: "
        f"{measurement['maximum_inward_deviation_mm']:.3f} mm"
    )

    print(
        f"Allowed tolerance: "
        f"{quality['maximum_allowed_dent_mm']:.3f} mm"
    )

    print(
        f"Defective points: "
        f"{measurement['defective_points']}/"
        f"{measurement['total_points']}"
    )

    print(
        f"Inspection decision: {quality['decision']}"
    )

    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
