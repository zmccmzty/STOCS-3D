from klampt.math import so3
import numpy as np
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

manipuland_fn = "data/manipulands/cube/cube.off"
terrain_fn = "data/environments/plane/cube.off"

gridres = 0.005
pcres = 0.005


# Set up manipuland
scale_x = 0.05
scale_y = 0.05
scale_z = 0.05
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
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]


# Task parameters
T_init = (so3.identity(),[0.5, 0.5, 0.0])

# Pushing
T_goal = (so3.identity(),[0.5, 0.7, 0.0])

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.025, 0.0, 0.04],
    [0.02, 0.0, 0.025],
    [0.03, 0.0, 0.025],
    [0.025, 0.0, 0.01],
]
manipulation_contact_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

#problem
problem = Problem(manipuland = manipuland,
                    manipuland_mass = 0.1,
                    manipuland_com=[0,0,0],
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
                    initial_pose_relaxation= 1e-3,  #tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
                    goal_pose_relaxation = 1e-3
                  )

# Optimizer parameters
oracle = SmoothingOracleParams(add_threshold=0.1,
                               remove_threshold=0.5,
                               translation_disturbances=[1e-2],
                               rotation_disturbances=[1e-2],
                               time_smoothing_step=1)


params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

