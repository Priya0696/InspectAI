
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d


def inspect_pipe(reference_path, scan_path, tolerance_mm=1.0):
    """
    Inspect a cylindrical pipe for inward surface defects.

    Assumptions:
    - Cylinder is aligned with the Z-axis.
    - Cylinder is centered at X=0, Y=0.
    - Reference and scan are already aligned.
    - Coordinates and tolerance are in millimeters.
    """

    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError(
            "Tolerance must be a positive finite number"
        )

    reference_path = Path(reference_path)
    scan_path = Path(scan_path)

    # Step 1: Load reference and scanned point clouds.
    reference = o3d.io.read_point_cloud(str(reference_path))
    scan = o3d.io.read_point_cloud(str(scan_path))

    if reference.is_empty() or scan.is_empty():
        raise ValueError(
            "Reference or scan point cloud is empty"
        )

    # Step 2: Convert point clouds into NumPy arrays.
    reference_points = np.asarray(reference.points)
    scan_points = np.asarray(scan.points)

    if not (
        np.isfinite(reference_points).all()
        and np.isfinite(scan_points).all()
    ):
        raise ValueError(
            "Point cloud contains non-finite coordinates"
        )

    # Step 3: Estimate nominal cylinder radius.
    reference_radii = np.linalg.norm(
        reference_points[:, :2],
        axis=1,
    )

    nominal_radius = float(
        np.median(reference_radii)
    )

    # Step 4: Calculate radial distances of scanned points.
    scan_radii = np.linalg.norm(
        scan_points[:, :2],
        axis=1,
    )

    # Step 5: Calculate inward deviations.
    inward_deviation = nominal_radius - scan_radii

    # Step 6: Identify defective points.
    defect_mask = inward_deviation > tolerance_mm

    defective_points = int(defect_mask.sum())
    total_points = len(scan_points)

    maximum_dent = float(
        max(0.0, inward_deviation.max())
    )

    defect_percentage = round(
        100 * defective_points / total_points,
        2,
    )

    # Step 7: Make quality-control decision.
    decision = (
        "FAIL" if defective_points > 0 else "PASS"
    )

    # Step 8: Generate inspection report.
    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "inspection_type": (
            "3D cylindrical surface inspection"
        ),

        "reference_file": str(reference_path),
        "scan_file": str(scan_path),

        "measurement": {
            "nominal_radius_mm": round(
                nominal_radius, 3
            ),
            "maximum_inward_deviation_mm": round(
                maximum_dent, 3
            ),
            "defective_points": defective_points,
            "total_points": total_points,
            "defective_sample_percentage": defect_percentage,
        },

        "quality_control": {
            "maximum_allowed_dent_mm": tolerance_mm,
            "decision": decision,
        },

        "method": {
            "registration": "Pre-aligned scan required",
            "defect_detection": (
                "Radial deviation from nominal cylinder"
            ),
            "assumptions": [
                "Cylinder aligned with Z-axis",
                "Cylinder centered at X=0, Y=0",
                "Scan and reference share coordinates",
                "Illustrative tolerance, not certified",
            ],
        },
    }
