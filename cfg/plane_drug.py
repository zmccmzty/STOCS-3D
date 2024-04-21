from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

environment_name = "plane"
manipuland_name = "drug"

manipuland_fn = "data/manipulands/drug/meshes/model.obj"
terrain_fn = "data/environments/plane/cube.off"

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
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]


# Task parameters
q_init = [8.81827229e-01, -4.70622799e-01, 1.54791655e-02, 2.55993819e-02]
x_init = [5.11917036e-01, 5.05103254e-01, 2.28624485e-02]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [0.82384476, -0.43072239, -0.17579866, -0.32380992]
x_goal = [0.45344041, 0.50510325, 0.0227302]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.0, 0.0, 0.1131],
    [0.02, 0.02, 0.1131],
    [0.02, -0.02, 0.1131],
    [-0.02, 0.02, 0.1131],
    [-0.02, -0.02, 0.1131],
]
manipulation_contact_normals = [
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
]
#problem
problem = Problem(manipuland = manipuland,
                    manipuland_mass = mass,
                    manipuland_com = com,
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
                    initial_pose_relaxation= 1e-2,  #tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
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

