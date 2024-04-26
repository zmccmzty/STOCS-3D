from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D


manipuland_fn = "data/manipulands/basket/model.obj"
terrain_fn = "data/environments/shelf/model.obj"

gridres = 0.005
pcres = 0.005

# Set up manipuland
# Used to calculate the center of mass of the manipuland
manipuland_mesh = trimesh.load_mesh(manipuland_fn)
mass = 0.1
com = list(manipuland_mesh.center_mass)
I = manipuland_mesh.moment_inertia*mass

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

# Set up environments
terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05], [0, 0, 0])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]


# Task parameters
q_init = [9.99977706e-01, -3.06899679e-03, 5.81738654e-03, -1.15199024e-03]
x_init = [0.4, 0.20324864, 0.0373261]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [9.99977706e-01, -3.06899679e-03, 5.81738654e-03, -1.15199024e-03]
x_goal = [0.19586671, 0.20324864, 0.0373261]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

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


#problem
problem = Problem(manipuland = manipuland,
                    manipuland_mass = mass,
                    manipuland_com=com,
                    manipuland_inertia = I,
                    environments = environments,
                    T_init = T_init,
                    T_goal = T_goal,
                    manipulation_contact_points=manipulation_contact_points,
                    manipulation_contact_normals=manipulation_contact_normals,
                    N= 6,              #trajectory discretization
                    time_step = 0.1,    #trajectory time step
                    x_bound = x_bound,
                    v_bound = v_bound,
                    w_max = np.pi,  #angular velocity bound
                    mu_env=1.0,  #friction
                    mu_mnp=1.0,  #friction
                    initial_pose_relaxation= 1e-2,  # tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
                    goal_pose_relaxation = 1e-2
                  )

# Optimizer parameters
oracle = SmoothingOracleParams(add_threshold=0.03,
                               remove_threshold=0.1,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1)


params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

