from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

manipuland_fn = "data/manipulands/koala/meshes/model.obj"
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
manipuland = PenetrationDepthGeometry(manipuland_g3d, gridres, pcres)

# Set up environments
terrain = Geometry3D()
terrain.loadFile(terrain_fn)
terrain.transform([1, 0, 0, 0, 1, 0, 0, 0, 0.25], [0, 0, -0.25])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]

# Task parameters
q_init = [1.0, 0.0, 0.0, 0.0]
x_init = [0.5, 0.4, 0.0]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [0.56253129, -0.82665503, 0.01, 0.01]
x_goal = [0.5, 0.5, 0.09544157]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.006962, -0.004155, 0.186040],
    [0.005816, -0.004939, 0.185172],
    [-0.001889, -0.001536, 0.184416],
]
manipulation_contact_normals = [
    [0.037155, 0.215888, -0.975711],
    [-0.410938, 0.079656, -0.908177],
    [0.079580, 0.523353, -0.848392],
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

# Optimizer parameters  -- note: no smoothing done here
oracle = SmoothingOracleParams(add_threshold=0.1,
                               remove_threshold=0.5,
                               translation_disturbances=[],
                               rotation_disturbances=[],
                               time_smoothing_step=0)

params = OptimizerParams(stocs_max_iter=20,
                         oracle_params=oracle)

