from klampt.math import so3
import numpy as np
from STOCS import Problem, OptimizerParams, SmoothingOracleParams

# import from semiinfinite, which is a package in the parent directory
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from semiinfinite.geometryopt import PenetrationDepthGeometry,Geometry3D

manipuland_fn = "data/manipulands/sphere/sphere.off"
terrain_fn = "data/environments/fractal/fractal.stl"

gridres = 0.005
pcres = 0.005

# Set up manipuland
scale_x, scale_y, scale_z = 0.05, 0.05, 0.05

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland_g3d.transform([scale_x, 0, 0, 0, scale_y, 0, 0, 0, scale_z], [0.0, 0.0, 0.0])
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

# Manipuland inertia
mass = 0.1
com = [0,0,0]
I = np.diag([1.66666667e-04]*3)

# Set up environments
terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05], [0, 0, 0])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]


# Task parameters
q_init = [0.95936369, 0.19777694, 0.05059618, -0.19479636]
x_init = [0.59475669, 0.60783364, 0.04642292]
T_init = (so3.from_quaternion(q_init), x_init)

# Pivoting
q_goal = [0.83955492, -0.49339478, -0.10207841, -0.2031973]
x_goal = [0.57104589, 0.67230184, 0.04346168]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [[0.0, -0.05, 0.0]]
manipulation_contact_normals = [[0, 1, 0]]

#problem
problem = Problem(manipuland = manipuland,
                    manipuland_mass = com,
                    manipuland_com= mass,
                    manipuland_inertia = I,
                    environments = environments,
                    T_init = T_init,
                    T_goal = T_goal,
                    manipulation_contact_points=manipulation_contact_points,
                    manipulation_contact_normals=manipulation_contact_normals,
                    N= 11,              #trajectory discretization
                    time_step = 0.1,    #trajectory time step
                    x_bound = x_bound,
                    v_bound = v_bound,
                    w_max = np.pi,  #angular velocity bound
                    mu_env=1.0,  #friction
                    mu_mnp=1.0,  #friction
                    manipuland_name="sphere",
                    environment_name="fractal",
                    initial_pose_relaxation= 0.1, #tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
                    goal_pose_relaxation = 0.1
                  )

# Optimizer parameters
oracle = SmoothingOracleParams(add_threshold=0.1,
                               remove_threshold=0.5,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1)


params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

