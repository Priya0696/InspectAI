
from pathlib import Path

import numpy as np
import open3d as o3d


# Reproducible synthetic industrial pipe
rng = np.random.default_rng(42)

radius = 20.0
height = 80.0
num_points = 10000

# Sample points on the outer cylindrical surface
theta = rng.uniform(0, 2 * np.pi, num_points)
z = rng.uniform(0, height, num_points)

x = radius * np.cos(theta)
y = radius * np.sin(theta)

points = np.column_stack((x, y, z))

# Simulate small measurement noise (millimeters)
noise = rng.normal(0, 0.1, points.shape)
points += noise

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.1, 0.6, 0.9])

# Save the point cloud
output = Path(
    "data/processed/pointclouds/reference_pipe.ply"
)
output.parent.mkdir(parents=True, exist_ok=True)

success = o3d.io.write_point_cloud(str(output), pcd)

if not success:
    raise RuntimeError("Failed to save point cloud")

print("\n=== INSPECTAI 3D DATA ===")
print("Object: Industrial pipe")
print("Points:", len(pcd.points))
print("Dimensions: millimeters")
print("Radius:", radius)
print("Height:", height)
print("Saved:", output)

# Open interactive 3D viewer
o3d.visualization.draw_geometries(
    [pcd],
    window_name="InspectAI - Reference Pipe",
    width=1000,
    height=700,
)
