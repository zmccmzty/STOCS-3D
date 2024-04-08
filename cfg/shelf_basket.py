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

environment_name = "shelf"
manipuland_name = "basket"

manipuland_fn = "data/manipulands/basket/model.obj"
terrain_fn = "data/environments/shelf/model.obj"

# Geometry in Trimesh
scale_x = 1
scale_y = 1
scale_z = 1
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
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0])

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland_g3d.transform([scale_x, 0, 0, 0, scale_y, 0, 0, 0, scale_z], [0.0, 0.0, 0.0])

environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

I = np.array([[0.00209333, -0.0, 0.0], [-0.0, 0.00269333, 0.0], [0.0, 0.0, 0.00156]])

# MPCC parameters
N = 6
comp_threshold = 1e-4
friction_dim = 6
max_mpcc_iter = 20

initial_pose_relaxation = 1e-2
goal_pose_relaxation = 1e-2
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
add_threshold = 5e-3
active_threshold = 1e-1
time_smoothing_step = 1
disturbances = [1e-2]

# Task parameters
q_init = [9.99977706e-01, -3.06899679e-03, 5.81738654e-03, -1.15199024e-03]
x_init = [0.4, 0.20324864, 0.0373261]

q_goal = [9.99977706e-01, -3.06899679e-03, 5.81738654e-03, -1.15199024e-03]
x_goal = [0.19586671, 0.20324864, 0.0373261]

task = "pushing"

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

mu_mnp = 1.0
mu_env = 1.0

manipulation_contact_points = [
    [0.178360, 0.003570, 0.194578],
    [0.178602, -0.029486, 0.207426],
    [0.177675, 0.017565, 0.199255],
]
manipulation_contact_normals = [
    [-0.999863, 0.016450, -0.001655],
    [-0.998074, 0.025674, 0.056467],
    [-0.999408, 0.002095, -0.034339],
]

manipuland_params = {
    "manipuland_name": manipuland_name,
    "manipuland_fn": manipuland_fn,
    "manipuland": manipuland,
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
