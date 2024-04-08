from klampt.math import so3
import numpy as np
import trimesh

# import from semiinfinite, which is a package in the parent directory
import sys
import os
import open3d as o3d

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from semiinfinite.geometryopt import *

environment_name = "fractal"
manipuland_name = "sphere"

manipuland_fn = "data/manipulands/sphere/sphere.off"
terrain_fn = "data/environments/fractal/fractal.stl"

# Load environment
distances_param = np.load(
    f"data/environments/{environment_name}/params.npy", allow_pickle=True
).item()
x1, x2, y1, y2, z1, z2, n1, n2, n3, h1, h2, h3 = (
    distances_param["x1"],
    distances_param["x2"],
    distances_param["y1"],
    distances_param["y2"],
    distances_param["z1"],
    distances_param["z2"],
    distances_param["n1"],
    distances_param["n2"],
    distances_param["n3"],
    distances_param["h1"],
    distances_param["h2"],
    distances_param["h3"],
)
grid_x = np.arange(n1 + 1) * h1 + x1
grid_y = np.arange(n2 + 1) * h2 + y1
grid_z = np.arange(n3 + 1) * h3 + z1

distances = np.load(f"data/environments/{environment_name}/distance_field.npy")
gradient_x = np.load(f"data/environments/{environment_name}/gradient_x_field.npy")
gradient_y = np.load(f"data/environments/{environment_name}/gradient_y_field.npy")
gradient_z = np.load(f"data/environments/{environment_name}/gradient_z_field.npy")

# Geometry in Trimesh
scale_x = 0.05
scale_y = 0.05
scale_z = 0.05
# Used to calculate the center of mass of the manipuland
manipuland_mesh = trimesh.load_mesh(manipuland_fn)
matrix = np.array(
    [[scale_x, 0, 0, 0], [0, scale_y, 0, 0], [0, 0, scale_z, 0], [0, 0, 0, 0]]
)
scaled_mesh = manipuland_mesh.apply_transform(matrix)
com = list(manipuland_mesh.center_mass)

# Geometry in Klampt
# Used to find closest point between the manipuland and the environment
gridres = 0.005
pcres = 0.005

terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05], [0, 0, 0])

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland_g3d.transform([scale_x, 0, 0, 0, scale_y, 0, 0, 0, scale_z], [0.0, 0.0, 0.0])

environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

object_mesh_o3d = o3d.io.read_triangle_mesh(manipuland_fn)
object_mesh_o3d.scale(scale_x, center=[0, 0, 0])

I = np.array(
    [
        [1.66666667e-04, -6.25000000e-05, -6.25000000e-05],
        [-6.25000000e-05, 1.66666667e-04, -6.25000000e-05],
        [-6.25000000e-05, -6.25000000e-05, 1.66666667e-04],
    ]
)

# MPCC parameters
N = 11
comp_threshold = 1e-4
friction_dim = 6
max_mpcc_iter = 20

initial_pose_relaxation = 1e-1
goal_pose_relaxation = 1e-1
major_feasibility_tolerance = 1e-4
major_optimality_tolerance = 1e-4
elastic_weight = 1000
time_step = 0.1
velocity_complementarity = True

# STOCS parameters
initialization = "linear"  # 'initial'/'linear'
stocs_max_iter = 20
assumption = "quasi_dynamic"  # 'quasi_static'/'quasi_dynamic'
force_on_off = False
use_alpha = False

# Oracle parameters
add_threshold = 1e-3
active_threshold = 1e-1
time_smoothing_step = 1
disturbances = [1e-2]

# Task parameters
q_init = [0.95936369, 0.19777694, 0.05059618, -0.19479636]
x_init = [0.59475669, 0.60783364, 0.04642292]

# Pivoting
q_goal = [0.83955492, -0.49339478, -0.10207841, -0.2031973]
x_goal = [0.57104589, 0.67230184, 0.04346168]
task = "pivoting"

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

mu_mnp = 1.0
mu_env = 1.0

manipulation_contact_points = [[0.0, -0.05, 0.0]]
manipulation_contact_normals = [[0, 1, 0]]

manipuland_params = {
    "manipuland_name": manipuland_name,
    "manipuland_fn": manipuland_fn,
    "manipuland": manipuland,
    "manipuland_g3d": manipuland_g3d,
    "manipuland_o3d": object_mesh_o3d,
    "scale_x": scale_x,
    "scale_y": scale_y,
    "scale_z": scale_z,
    "com": com,
    "m": 0.1,
    "I": I,
}

environment_params = {
    "environment_name": environment_name,
    "distances": distances,
    "gradient_x": gradient_x,
    "gradient_y": gradient_y,
    "gradient_z": gradient_z,
    "environments": environments,
    "distances_param": distances_param,
}

optimization_params = {
    "comp_threshold": comp_threshold,
    "friction_dim": friction_dim,
    "mu_mnp": mu_mnp,
    "mu_env": mu_env,
    "N": N,
    "stocs_max_iter": stocs_max_iter,
    "velocity_complementarity": velocity_complementarity,
    "initial_pose_relaxation": initial_pose_relaxation,
    "goal_pose_relaxation": goal_pose_relaxation,
    "max_mpcc_iter": max_mpcc_iter,
    "major_feasibility_tolerance": major_feasibility_tolerance,
    "major_optimality_tolerance": major_optimality_tolerance,
    "time_step": time_step,
    "initialization": initialization,
    "elastic_weight": elastic_weight,
    "disturbances": disturbances,
    "add_threshold": add_threshold,
    "active_threshold": active_threshold,
    "time_smoothing_step": time_smoothing_step,
    "assumption": assumption,
    "force_on_off": force_on_off,
    "use_alpha": use_alpha,
}
task_params = {
    "q_init": q_init,
    "x_init": x_init,
    "q_goal": q_goal,
    "x_goal": x_goal,
    "task_name": task,
    "manipulation_contact_points": manipulation_contact_points,
    "manipulation_contact_normals": manipulation_contact_normals,
    "x_bound": x_bound,
    "v_bound": v_bound,
}

params = {
    "manipuland_params": manipuland_params,
    "environment_params": environment_params,
    "optimization_params": optimization_params,
    "task_params": task_params,
}
