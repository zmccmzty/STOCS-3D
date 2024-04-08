# Simultaneous Trajectory Optimization and Contact Selection in 3D

## File structure

```
├── README.md                 This file
├── data                      Object files for running the example code
├── cfg                       Configuration files for running the example code
├── MPCC.py                   Mathematical Program with Complementarity Constraint
├── STOCS.py                  Simultaneous Trajectory Optimization and Contact Selection
├── plot_result.py            A script to visualize the planned trajectory
└── setup_environment.sh      A script to install all the dependencies
```
## Create venv and install dependencies
> bash setup_environment.sh

## Download the pre-computed Signed Distance Fields (SDF)
[https://drive.google.com/file/d/1bMg7aNiUAnojUgs8koyhwdJXODThaNSN/view?usp=drive_link]

Download the pre-computed SDFs for the environments used in the demos, and place them in the folder data/environments.

## Running demos

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

## Visualize planned trajectory
> python STOCS.py --result_path=results/plane_koala_STOCS/result.npy --environment=plane --manipuland=koala
