from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D


manipuland_fn = "data/manipulands/mustard/textured.obj"
terrain_fn = "data/environments/plane/cube.off"

#resolution for geometry discretization.  The environment SDF will be used while the a point cloud for the manipuland will be used.
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
#manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)
manipuland = PenetrationDepthGeometry(manipuland_g3d, None, pcres)

# Set up environments
terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])
environments = [PenetrationDepthGeometry(terrain, gridres, None)]
#environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]


# Task parameters
T_init=([
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
    [0.5, 0.5, 0.05218423])

T_goal = (so3.mul(so3.from_axis_angle(([1, 0, 0], np.pi / 2)),T_init[0]),[0.5, -0.05218423 + 0.5, 0.0031377])

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

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
                    initial_pose_relaxation= 1e-2,  #tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
                    goal_pose_relaxation = 1e-2
                  )

# Optimizer parameters 
oracle = SmoothingOracleParams(add_threshold=0.01,
                               remove_threshold=0.1,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1)

params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

