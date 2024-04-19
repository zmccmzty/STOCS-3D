# Simultaneous Trajectory Optimization and Contact Selection in 3D

This library implements the 

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

First, you must download the pre-computed SDFs for the environments used in the demos from the following link, and place them in the folder `data/environments`.
[https://drive.google.com/file/d/1bMg7aNiUAnojUgs8koyhwdJXODThaNSN/view?usp=drive_link]

Then you can run STOCS on the example environments and objects as follows:

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

Results will be placed in the `tmp` directory.

## Visualize planned trajectory
> python STOCS.py --result_path=results/plane_koala_STOCS/result.npy --environment=plane --manipuland=koala

## Configuring the solver

### Defining the object and environment

You will need an SDF of your environment and a point cloud of your object.

### Defining robot contact parameters

### Defining solver parameters
