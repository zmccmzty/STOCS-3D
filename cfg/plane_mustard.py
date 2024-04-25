from klampt.math import so3
import numpy as np
import trimesh

# import from semiinfinite, which is a package in the parent directory
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from semiinfinite.geometryopt import *

environment_name = "plane"
manipuland_name = "mustard"

manipuland_fn = "data/manipulands/mustard/textured.obj"
terrain_fn = "data/environments/plane/cube.off"



# Geometry in Trimesh
scale_x = 1.0
scale_y = 1.0
scale_z = 1.0
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
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland_g3d.transform([scale_x, 0, 0, 0, scale_y, 0, 0, 0, scale_z], [0.0, 0.0, 0.0])

environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

I = np.array(
    [
        [1.63398693e-04, -3.47882886e-05, 1.13537268e-04],
        [-3.47882886e-05, 3.26797386e-04, 1.74288078e-04],
        [1.13537268e-04, 1.74288078e-04, 1.63398693e-04],
    ]
)

# MPCC parameters
N = 11
comp_threshold = 1e-4
friction_dim = 6
max_mpcc_iter = 20

initial_pose_relaxation = 1e-2
goal_pose_relaxation = 1e-2
major_feasibility_tolerance = 1e-6
major_optimality_tolerance = 1e-4
elastic_weight = 1000
time_step = 0.1
velocity_complementarity = True

# STOCS parameters
initialization = "linear"  # 'initial'
stocs_max_iter = 20
assumption = "quasi_dynamic"  # 'quasi_static'/'quasi_dynamic'
force_on_off = False
use_alpha = False

# Oracle parameters
add_threshold = 1e-3
active_threshold = 1e-2
time_smoothing_step = 1
disturbances = [1e-2]

# Task parameters
q_init = so3.quaternion(
    [
        0.41596866,
        0.00322376,
        0.90937323,
        0.90937895,
        -0.00147461,
        -0.41596605,
        0.0,
        0.99999372,
        -0.00354501,
    ]
)
x_init = [0.5, 0.5, 0.05218423]

q_goal = so3.quaternion(
    so3.mul(
        so3.from_axis_angle(([1, 0, 0], np.pi / 2)),
        [
            0.41596866,
            0.00322376,
            0.90937323,
            0.90937895,
            -0.00147461,
            -0.41596605,
            0.0,
            0.99999372,
            -0.00354501,
        ],
    )
)
x_goal = [0.5, -0.05218423 + 0.5, 0.0031377]
task = "pivoting"

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

mu_mnp = 1.0
mu_env = 1.0

manipulation_contact_points = [
    [-0.011, -0.017, 0.187],
    [-0.010, -0.025, 0.187],
    [-0.018, -0.020, 0.187],
    [-0.015, -0.026, 0.187],
    [-0.012, -0.022, 0.187],
]
manipulation_contact_normals = [
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
]

manipuland_params = {
    "manipuland_name": manipuland_name,
    "manipuland_fn": manipuland_fn,
    "manipuland": manipuland,
    "manipuland_g3d": manipuland_g3d,
    "scale_x": scale_x,
    "scale_y": scale_y,
    "scale_z": scale_z,
    "com": com,
    "m": 0.1,
    "I": I,
}

environment_params = {
    "environment_name": environment_name,
    "environments": environments,
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
