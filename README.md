# Simultaneous Trajectory Optimization and Contact Selection in 3D

This library implements the STOCS algorithm to optimize trajectories for general 3D objects in contact with an environment and a robot manipulator.  STOCS is able to take advantage of **all** possible contact points between the object and environment during optimization using an infinite programming approach. 


## File structure

```
├── README.md                 This file
├── data                      Object files for running the example code
├── cfg                       Configuration files for running the example code
├── MPCC.py                   Mathematical Program with Complementarity Constraint
├── STOCS.py                  Simultaneous Trajectory Optimization and Contact Selection
├── plot_result.py            A script to visualize the planned trajectory
└── requirements.txt          Python dependencies file
```
## Installing dependencies

Install python dependencies as follows
> python -m pip install -r requirements.txt

(To install system-wide, use sudo)

To install in a virtual environment, you may use the following:

> python3 -m venv stocs_env
> stocs_env/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt


## Running demos

You can run STOCS on the example environments and objects as follows:

> python STOCS.py --environment=plane --manipuland=koala

> python STOCS.py --environment=plane --manipuland=shoe

> python STOCS.py --environment=plane --manipuland=drug

> python STOCS.py --environment=plane --manipuland=cube

> python STOCS.py --environment=plane --manipuland=sphere

> python STOCS.py --environment=plane --manipuland=mustard

> python STOCS.py --environment=plane --manipuland=tool

> python STOCS.py --environment=plate --manipuland=plate

> python STOCS.py --environment=shelf --manipuland=basket

> python STOCS.py --environment=sofa --manipuland=pillow

> python STOCS.py --environment=fractal --manipuland=koala

> python STOCS.py --environment=fractal --manipuland=sphere

Results will be placed in the `results` directory.  Debugging information of the SNOPT algorithm will be written to `tmp/debug.txt`.

## Visualize planned trajectory
> python STOCS.py --result_path=results/plane_koala_STOCS/result.npy --environment=plane --manipuland=koala

## Configuring the solver

### Defining the object and environment

You will need a triangle mesh defining your environment and object (manipuland).  (Minimally, you could get away with a point cloud of the manipuland, but this would work a little differently than the examples, all of which use meshes.)

To specify the manipuland, you will create a dictionary with the following structure:

```python
manipuland_params = {
    "manipuland_name": manipuland_name, # an identifier
    "manipuland_fn": manipuland_fn,     # the mesh file name, used for visualization
    "manipuland": manipuland,           # a PenetrationDepthGeometry created from the mesh
    "scale_x": scale_x,                 # the x scale applied to the mesh file to get the geometry 
    "scale_y": scale_y,                 # the y scale applied to the mesh file to get the geometry
    "scale_z": scale_z,                 # the z scale applied to the mesh file to get the geometry
    "com": com,                         # the center of mass in the geometry's local frame
    "m": 0.1,                           # the manipuland's mass
    "I": I,                             # the manipuland's inertia
}
```

To specify the environment, you will create a dictionary with the following structure:

```python
environment_params = {
    "environment_name": environment_name,   # an identifier
    "environments": environments,           # a list of PenetrationDepthGeometry 
}
```

### Defining robot contact parameters and terminal conditions

```python
task_params = {
    "q_init": q_init,           # quaternion representation of initial manipuland rotation
    "x_init": x_init,           # quaternion representation of initial manipuland translation
    "q_goal": q_goal,           # quaternion representation of goal manipuland rotation
    "x_goal": x_goal,           # quaternion representation of goal manipuland translation
    "task_name": task,          # identifier of task (not used)
    "manipulation_contact_points": manipulation_contact_points,   # list of points on the manipuland that the robot is touching. expressed in local frame. 
    "manipulation_contact_normals": manipulation_contact_normals, # list of normals corresponding to points on the manipuland that the robot is touching.  Pointing inward to manipuland, and expressed in local frame.
    "x_bound": x_bound,         # bounds for the translation of the manipuland
    "v_bound": v_bound,         # bounds for the velocity of the manipuland
}
```

### Defining solver parameters
