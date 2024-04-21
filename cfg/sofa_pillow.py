from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D


manipuland_fn = "data/manipulands/pillow/model.obj"
terrain_fn = "data/environments/sofa/model.obj"

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
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]

# Task parameters
q_init = [0.93201937, 0.01132791, -0.01573812, 0.36188933]
x_init = [0.29673823, 0.24654253, 0.24508384]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [0.66427177, 0.27132799, -0.65815855, 0.22792862]
x_goal = [0.22174031, 0.18297616, 0.30597341]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.096089, -0.018228, -0.003647],
    [0.095825, 0.023243, -0.004128],
    [0.094938, 0.004916, -0.010575],
    [0.087571, -0.000795, -0.020456],
]
manipulation_contact_normals = [
    [-0.998571, -0.026387, 0.046470],
    [-0.975133, 0.212204, 0.063921],
    [-0.988077, -0.107921, 0.109803],
    [-0.92169493, 0.046842, 0.38507697],
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
                    N= 11,              #trajectory discretization
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
oracle = SmoothingOracleParams(add_threshold=0.1,
                               remove_threshold=0.5,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1)


params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)
