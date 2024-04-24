from STOCS import TimestepState,TrajectoryState,Problem
from klampt.math import so3,se3,vectorops
import numpy as np

def plot_index_points_o3d(manipuland_o3d, state : TimestepState):
    import open3d as o3d
    xyz = np.zeros((len(state.index_set),3))
    for i,index_pt in enumerate(state.index_set):
        xyz[i,:] = index_pt
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coord.scale(0.1, center=[0,0,0])
    T = se3.ndarray(state.pose())
    manipuland_o3d.transform(T)
    pcd.transform(T)
    coord.translate(T[1])
    #o3d.visualization.draw_geometries([object_mesh_o3d,pcd,coord])
    o3d.visualization.draw_geometries([manipuland_o3d,pcd])
    manipuland_o3d.transform(np.linalg.inv(T))

def plot_problem_klampt(problem : Problem, show=True):
    """Shows just a problem setup in Klampt visualization"""
    import klampt
    from klampt import vis
    from klampt.vis import gldraw
    world = klampt.WorldModel()
    cone_geom = klampt.Geometry3D()
    cone_geom.loadFile("data/plot/cone.stl")
    manipulator_size = 1
    vis.add("x_bounds",problem.x_bound)
    def draw_bounds(bb,color=(1,1,1)):
        from OpenGL.GL import glColor3f
        glColor3f(*color)
        gldraw.box(bb[0],bb[1],lighting=False,filled=False)
    vis.setDrawFunc("x_bounds",lambda data:draw_bounds(data,(1,1,1)))

    vis.add("init_bounds",[vectorops.add(problem.T_init[1],[-problem.initial_pose_relaxation]*3),vectorops.add(problem.T_init[1],[problem.initial_pose_relaxation]*3)])
    vis.setDrawFunc("init_bounds",lambda data:draw_bounds(data,(0,1,0)))

    vis.add("goal_bounds",[vectorops.add(problem.T_goal[1],[-problem.goal_pose_relaxation]*3),vectorops.add(problem.T_goal[1],[problem.goal_pose_relaxation]*3)])
    vis.setDrawFunc("goal_bounds",lambda data:draw_bounds(data,(1,0,0)))

    for i,e in enumerate(problem.environments):
        world.makeTerrain("terrain_"+str(i))
        world.terrain(i).geometry().set(problem.environments[i].geom)
    
    # Plot the object
    vis.add("init_pose",problem.T_init)
    vis.add("goal_pose",problem.T_goal)
    obj_0 = world.makeRigidObject("manipuland_init")
    obj_0.geometry().set(problem.manipuland.geom)
    obj_0.setTransform(*problem.T_init)
    obj_0.appearance().setColor(0,1,0,0.5)
    obj_0.appearance().setSilhouette(0.001)

    obj_1 = world.makeRigidObject("manipuland_goal")
    obj_1.geometry().set(problem.manipuland.geom)
    obj_1.setTransform(*problem.T_goal)
    obj_1.appearance().setColor(1,0,0,0.5)
    obj_1.appearance().setSilhouette(0.001)

    # Plot manipulator's contact locations
    point_world = np.zeros(3)
    point_local = np.zeros(3)
    nc = len(problem.manipulation_contact_points)
    for j in range(nc):
        point_world += np.array(se3.apply(problem.T_init,problem.manipulation_contact_points[j])) / nc
        point_local += np.array(problem.manipulation_contact_points[j]) / nc

    manipulator_0 = world.makeRigidObject("manipulator_start")
    manipulator_0.geometry().set(cone_geom)
    manipulator_0.geometry().transform([manipulator_size,0,0,0,manipulator_size,0,0,0,manipulator_size],[0.0,0.0,0.0])

    problem.manipuland.setTransform((so3.identity(),[0,0,0]))

    normal_direction = problem.manipuland.normal(point_local)
    normal_direction = so3.apply(problem.T_init[0],normal_direction)

    manipulator_0.setTransform(so3.canonical(normal_direction),point_world)
    manipulator_0.appearance().setColor(1,0,0,1.0)
    manipulator_0.appearance().setSilhouette(0.001)
    
    vis.add("world",world)
    if show:
        vis.run()

def plot_trajectory_klampt(traj : TrajectoryState, problem : Problem, show=True):
    import klampt
    from klampt import vis
    world = klampt.WorldModel()
    cone_geom = klampt.Geometry3D()
    cone_geom.loadFile("data/plot/cone.stl")
    plot_shift = 0.0
    for i in range(0,problem.N):
        # Plot the object
        obj_i = world.makeRigidObject("object_"+str(i))
        obj_i.geometry().set(problem.manipuland.geom)
        Ti = traj.states[i].pose()
        Ti = (Ti[0],vectorops.add(Ti[1],[plot_shift*i,0,0]))

        obj_i.setTransform(*Ti)
        obj_i.appearance().setColor(0,0.1*i,1,0.2)
        obj_i.appearance().setSilhouette(0.001)

        # Plot manipulator's contact locations
        point_world = np.zeros(3)
        point_local = np.zeros(3)
        nc = len(problem.manipulation_contact_points)
        for j in range(nc):
            point_world += np.array(se3.apply(Ti,problem.manipulation_contact_points[j])) / nc
            point_local += np.array(problem.manipulation_contact_points[j]) / nc

        manipulator_size = 1
        manipulator_i = world.makeRigidObject("manipulator_"+str(i))
        manipulator_i.geometry().set(cone_geom)
        manipulator_i.geometry().transform([manipulator_size,0,0,0,manipulator_size,0,0,0,manipulator_size],[0.0,0.0,0.0])

        problem.manipuland.setTransform((so3.identity(),[0,0,0]))

        normal_direction = problem.manipuland.normal(point_local)
        normal_direction = so3.apply(Ti[0],normal_direction)

        #manipulator_i.setTransform(so3.canonical(normal_direction),point_world)
        manipulator_i.appearance().setColor(1,0,0,1.0)
        manipulator_i.appearance().setSilhouette(0.001)
        point_plot = [point_world[0]+plot_shift*i,point_world[1],point_world[2]]
        manipulator_i.setTransform(so3.canonical(normal_direction),point_plot)

    vis.add("world",world)
    if show:
        vis.run()
        vis.kill()
        vis.scene().clear()
