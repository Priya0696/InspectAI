
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

if reference.is_empty() or defective.is_empty():
    raise RuntimeError("One or both point clouds are empty")

# Create a rotated and translated scan to simulate
# a component positioned differently from the reference.
scan = o3d.geometry.PointCloud(defective)

rotation = o3d.geometry.get_rotation_matrix_from_xyz(
    (0.04, -0.06, 0.08)
)

scan.rotate(rotation, center=(0, 0, 0))
scan.translate((2.0, -1.5, 1.0))

print("\n=== BEFORE REGISTRATION ===")
print("Reference points:", len(reference.points))
print("Scan points:", len(scan.points))

# ICP needs an initial alignment close enough to converge.
# We know the approximate initial pose in this simulation.
initial_transform = np.eye(4)
initial_transform[:3, :3] = rotation.T
initial_transform[:3, 3] = (
    -rotation.T @ np.array([2.0, -1.5, 1.0])
)

threshold = 5.0  # millimeters

result = o3d.pipelines.registration.registration_icp(
    scan,
    reference,
    threshold,
    initial_transform,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=100
    ),
)

print("\n=== ICP RESULTS ===")
print("Fitness:", result.fitness)
print("Inlier RMSE (mm):", result.inlier_rmse)
print("Transformation:")
print(result.transformation)

# Apply the estimated transformation
scan.transform(result.transformation)

reference.paint_uniform_color([0.1, 0.6, 1.0])
scan.paint_uniform_color([1.0, 0.3, 0.2])

output = DATA_DIR / "registered_defective_pipe.ply"
o3d.io.write_point_cloud(str(output), scan)

print("\nSaved:", output)

o3d.visualization.draw_geometries(
    [reference, scan],
    window_name="InspectAI - ICP Registration",
    width=1100,
    height=750,
)
