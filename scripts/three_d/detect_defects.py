
from pathlib import Path

import numpy as np
import open3d as o3d


DATA_DIR = Path("data/processed/pointclouds")

reference = o3d.io.read_point_cloud(
    str(DATA_DIR / "reference_pipe.ply")
)

defective = o3d.io.read_point_cloud(
    str(DATA_DIR / "defective_pipe.ply")
)

reference_points = np.asarray(reference.points)
defective_points = np.asarray(defective.points)

if len(reference_points) != len(defective_points):
    raise ValueError("Point counts do not match")

# The synthetic data preserves point correspondence.
# Compare radial distances to measure inward deformation.
reference_radius = np.linalg.norm(
    reference_points[:, :2], axis=1
)

defective_radius = np.linalg.norm(
    defective_points[:, :2], axis=1
)

radial_deviation = reference_radius - defective_radius

# Detect inward dents greater than 1 mm.
threshold_mm = 1.0
defect_mask = radial_deviation > threshold_mm

print("\n=== 3D DEFECT DETECTION ===")
print("Total points:", len(defective_points))
print("Defective points:", int(defect_mask.sum()))
print(
    "Maximum dent depth (mm):",
    round(float(radial_deviation.max()), 3),
)
print(
    "Mean detected dent depth (mm):",
    round(float(radial_deviation[defect_mask].mean()), 3)
    if defect_mask.any() else 0,
)
print(
    "Defective surface samples (%):",
    round(100 * defect_mask.mean(), 2),
)

# Color the defective region red and normal points blue.
colors = np.tile(
    np.array([0.1, 0.6, 0.9]),
    (len(defective_points), 1),
)

colors[defect_mask] = [1.0, 0.0, 0.0]

defective.colors = o3d.utility.Vector3dVector(colors)

output = DATA_DIR / "detected_defects.ply"

if not o3d.io.write_point_cloud(str(output), defective):
    raise RuntimeError("Failed to save defect visualization")

print("Saved:", output)

o3d.visualization.draw_geometries(
    [defective],
    window_name="InspectAI - 3D Defect Detection",
    width=1100,
    height=750,
)
