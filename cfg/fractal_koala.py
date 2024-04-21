from klampt.math import so3
import numpy as np
import trimesh
from STOCS import Problem, OptimizerParams, SmoothingOracleParams, PenetrationDepthGeometry, Geometry3D

manipuland_fn = "data/manipulands/koala/meshes/model.obj"
terrain_fn = "data/environments/fractal/fractal.stl"

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
terrain.transform([0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05], [0, 0, 0])
environments = [PenetrationDepthGeometry(terrain, gridres, pcres)]

# Task parameters
q_init = [0.51126341, -0.85217643, -0.01492536, 0.11038458]
x_init = [0.40606108, 0.42948159, 0.08353624]
T_init = (so3.from_quaternion(q_init), x_init)

q_goal = [0.31060729, -0.61289434, -0.3736227, -0.62312897]
x_goal = [0.24210417, 0.33033184, 0.08472059]
T_goal = (so3.from_quaternion(q_goal), x_goal)

x_bound = [[0, 0, 0], [1, 1, 1]]
v_bound = [[-1, -1, -1], [1, 1, 1]]

manipulation_contact_points = [
    [0.058710, -0.001711, 0.060868],
    [0.054381, 0.003724, 0.045747],
    [0.057530, -0.004312, 0.061971],
]
manipulation_contact_normals = [
    [-0.92959988, 0.14932745, 0.33696494],
    [-0.93911024, -0.14896772, 0.30964589],
    [-0.98456283, 0.16412737, -0.06081311],
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
                    initial_pose_relaxation= 0.1,  # tolerance parameters -- lets the initial and goal pose wiggle a bit to attain feasibility
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

