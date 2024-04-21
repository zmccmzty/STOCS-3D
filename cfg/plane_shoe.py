from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

manipuland_fn = "data/manipulands/shoe/meshes/model.obj"
terrain_fn = "data/environments/plane/cube.off"


#resolution for geometry discretization.  The environment SDF will be used while the a point cloud for the manipuland will be used.
gridres = 0.005
pcres = 0.005

# Set up manipuland
scale_x = 1.0
scale_y = 1.0
scale_z = 1.0
# Used to calculate the center of mass of the manipuland
manipuland_mesh = trimesh.load_mesh(manipuland_fn)
matrix = np.diag([scale_x,scale_y,scale_z,1])
scaled_mesh = manipuland_mesh.apply_transform(matrix)
mass = 0.1
com = list(manipuland_mesh.center_mass)
I = manipuland_mesh.moment_inertia*mass

manipuland_g3d = Geometry3D()
manipuland_g3d.loadFile(manipuland_fn)
manipuland_g3d.transform([scale_x, 0, 0, 0, scale_y, 0, 0, 0, scale_z], [0.0, 0.0, 0.0])
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

# Set up environments
terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]



# Task parameters
q_init = [0.9999457, 0.00231579, -0.0020749, 0.00994673]
x_init = [0.51, 0.51, 0.00065307]
T_init = (so3.from_quaternion(q_init), x_init)

# Pushing
q_goal = [0.9999457, 0.00231579, -0.0020749, 0.00994673]
x_goal = [0.2, 0.51, 0.00065307]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.040351, 0.025581, 0.069064],
    [0.040524, 0.016156, 0.064304],
    [0.034509, 0.037068, 0.045343],
]
manipulation_contact_normals = [
    [-0.97812056, 0.07639362, -0.19350502],
    [-0.99510767, 0.05839668, -0.07969032],
    [-0.96353969, 0.03947309, 0.26463775],
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
                               time_smoothing_step=1,
                               duplicate_detection_threshold=5e-3)

params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

