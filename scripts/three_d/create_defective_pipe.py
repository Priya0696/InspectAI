
from pathlib import Path

import numpy as np
import open3d as o3d


# Load the existing reference pipe
reference_path = Path(
    "data/processed/pointclouds/reference_pipe.ply"
)

if not reference_path.exists():
    raise FileNotFoundError(
        f"Reference point cloud missing: {reference_path}"
    )

reference = o3d.io.read_point_cloud(str(reference_path))

points = np.asarray(reference.points).copy()

print("Reference points:", len(points))

# Simulate a localized dent in the pipe
dent_angle = np.pi / 4
dent_height = 40.0

angles = np.arctan2(points[:, 1], points[:, 0])

# Handle circular angular differences correctly
angle_difference = np.arctan2(
    np.sin(angles - dent_angle),
    np.cos(angles - dent_angle),
)

height_difference = points[:, 2] - dent_height

# Gaussian-shaped dent
dent_strength = 4.0 * np.exp(
    -0.5 * (
        (angle_difference / 0.25) ** 2
        + (height_difference / 8.0) ** 2
    )
)

# Move points inward toward the cylinder axis
radial_distance = np.linalg.norm(points[:, :2], axis=1)

radial_direction = (
    points[:, :2] / radial_distance[:, None]
)

points[:, :2] -= (
    dent_strength[:, None] * radial_direction
)

# Create defective point cloud
defective = o3d.geometry.PointCloud()
defective.points = o3d.utility.Vector3dVector(points)

reference.paint_uniform_color([0.2, 0.7, 1.0])
defective.paint_uniform_color([1.0, 0.3, 0.2])

output = Path(
    "data/processed/pointclouds/defective_pipe.ply"
)

o3d.io.write_point_cloud(str(output), defective)

print("\n=== DEFECT SIMULATION ===")
print("Defect type: Localized dent")
print("Maximum intended dent depth: 4 mm")
print("Points:", len(defective.points))
print("Saved:", output)

# Shift the defective pipe for side-by-side viewing
defective.translate((55, 0, 0))

o3d.visualization.draw_geometries(
    [reference, defective],
    window_name="InspectAI - Reference vs Defective Pipe",
    width=1100,
    height=750,
)
