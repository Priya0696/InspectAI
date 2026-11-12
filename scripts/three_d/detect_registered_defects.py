
from pathlib import Path

import numpy as np
import open3d as o3d


DATA_DIR = Path("data/processed/pointclouds")

reference = o3d.io.read_point_cloud(
    str(DATA_DIR / "reference_pipe.ply")
)

scan = o3d.io.read_point_cloud(
    str(DATA_DIR / "registered_defective_pipe.ply")
)

if reference.is_empty() or scan.is_empty():
    raise RuntimeError("Reference or registered scan is empty")

reference_points = np.asarray(reference.points)
scan_points = np.asarray(scan.points)

# Estimate the reference cylinder radius.
reference_radii = np.linalg.norm(
    reference_points[:, :2], axis=1
)
nominal_radius = float(np.median(reference_radii))

# Measure inward deviation from the cylinder surface.
scan_radii = np.linalg.norm(scan_points[:, :2], axis=1)
inward_deviation = nominal_radius - scan_radii

threshold_mm = 1.0
defect_mask = inward_deviation > threshold_mm

print("\n=== REGISTERED 3D DEFECT DETECTION ===")
print(f"Reference radius: {nominal_radius:.3f} mm")
print(f"Total scan points: {len(scan_points)}")
print(f"Defective points: {defect_mask.sum()}")
print(f"Defective samples: {defect_mask.mean() * 100:.2f}%")

if defect_mask.any():
    print(
        f"Maximum detected depth: "
        f"{inward_deviation[defect_mask].max():.3f} mm"
    )
    print(
        f"Mean detected depth: "
        f"{inward_deviation[defect_mask].mean():.3f} mm"
    )
else:
    print("No defects detected above threshold")

# Blue = normal, red = defective.
colors = np.tile(
    [0.1, 0.6, 0.9],
    (len(scan_points), 1),
)
colors[defect_mask] = [1.0, 0.0, 0.0]

scan.colors = o3d.utility.Vector3dVector(colors)

output_path = DATA_DIR / "registered_defects_detected.ply"

if not o3d.io.write_point_cloud(str(output_path), scan):
    raise RuntimeError("Failed to save output point cloud")

print(f"Saved: {output_path}")

o3d.visualization.draw_geometries(
    [scan],
    window_name="InspectAI - Registered 3D Defects",
    width=1100,
    height=750,
)
