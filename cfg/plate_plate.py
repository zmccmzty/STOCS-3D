from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

manipuland_fn = "data/manipulands/plate/model.obj"
terrain_fn = "data/environments/plate/model.obj"

gridres = 0.002
pcres = 0.002

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



# # Task parameters
q_init = [0.99870983, -0.04944309, -0.00583609, -0.01]
x_init = [0.15671865, 0.13485141, 0.01827428]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [0.99996641, -0.00112441, 0.00560948, 0.00586958]
x_goal = [0.15347964, 0.15272133, 0.01164616]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.003046, -0.101541, 0.026404],
    [-0.026470, -0.098431, 0.020578],
    [0.008961, -0.098059, 0.021053],
]
manipulation_contact_normals = [
    [-0.018017, 0.972181, 0.233538],
    [-0.031354, 0.772119, 0.634704],
    [-0.237138, 0.852140, 0.466502],
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
                    initial_pose_relaxation= 5e-3,  #tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
                    goal_pose_relaxation = 5e-3
                  )

# Optimizer parameters 
oracle = SmoothingOracleParams(add_threshold=0.1,
                               remove_threshold=0.3,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1,
                               duplicate_detection_threshold=5e-3)

params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

