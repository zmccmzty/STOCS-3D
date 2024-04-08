from icecream import ic
import numpy as np
import sys
import time
import copy
import math
from functools import partial
from klampt.math import vectorops,so3,se3
from MPCC import MPCC
import trimesh
import open3d as o3d
import os
from klampt import vis
from semiinfinite.geometryopt import *

import argparse
import importlib
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, required=True)

args = parser.parse_args()
result_path = args.result_path

environment,manipuland = result_path.split('/')[-2].split('_')[0],result_path.split('/')[-2].split('_')[1]
module_name = f"cfg.{environment}_{manipuland}"
module = importlib.import_module(module_name)
params = getattr(module, 'params', None)

res = np.load(result_path, allow_pickle=True).item()
iter = res['total_iter']
q_sol = res[f'iter_{iter}']['q']
x_sol = res[f'iter_{iter}']['x']

distances_param = params['environment_params']['distances_param']
x1, x2, y1, y2, z1, z2, n1, n2, n3, h1, h2, h3 = (
    distances_param["x1"],
    distances_param["x2"],
    distances_param["y1"],
    distances_param["y2"],
    distances_param["z1"],
    distances_param["z2"],
    distances_param["n1"],
    distances_param["n2"],
    distances_param["n3"],
    distances_param["h1"],
    distances_param["h2"],
    distances_param["h3"],
)

h1, h2, h3 = (x2 - x1) / n1, (y2 - y1) / n2, (z2 - z1) / n3 # grid spacing
grid_x = np.arange(n1 + 1) * h1 + x1 # grid points in x direction
grid_y = np.arange(n2 + 1) * h2 + y1 # grid points in y direction
grid_z = np.arange(n3 + 1) * h3 + z1 # grid points in z direction

gradient_x = params['environment_params']['gradient_x']
gradient_y = params['environment_params']['gradient_y']
gradient_z = params['environment_params']['gradient_z']
distances = params['environment_params']['distances']

def distance_function(x,y,z):
    return trilinear_interpolation(grid_x, grid_y, grid_z, distances, x, y, z)

def gradient_x_function(x,y,z):
    return trilinear_interpolation(grid_x, grid_y, grid_z, gradient_x, x, y, z)

def gradient_y_function(x,y,z):
    return trilinear_interpolation(grid_x, grid_y, grid_z, gradient_y, x, y, z)

def gradient_z_function(x,y,z):
    return trilinear_interpolation(grid_x, grid_y, grid_z, gradient_z, x, y, z)

import klampt
vis.init()
world = klampt.WorldModel()
vis.setBackgroundColor(1,1,1)

for i,environment in enumerate(params['environment_params']['environments']):
    vis.add("env"+str(i),params['environment_params']['environments'][i].grid,hide_label=True)
    vis.setColor("env"+str(i),1,0.5,0,1.0)

scale_x = params['manipuland_params']['scale_x']
scale_y = params['manipuland_params']['scale_y']
scale_z = params['manipuland_params']['scale_z']

contact_points = res['param']['task_params']['manipulation_contact_points']
index_set = res[f'iter_{iter}']['index_set']
env_contact_points = res[f'iter_{1}']['index_set']
friction_dim = res['param']['optimization_params']['friction_dim']
q_init = res['param']['task_params']['q_init']
x_init = res['param']['task_params']['x_init']
q_goal = res['param']['task_params']['q_goal']
x_goal = res['param']['task_params']['x_goal']
N = res['param']['optimization_params']['N']

plot_shift = 0.0
for i in range(0,params['optimization_params']['N']):

    # Plot the object
    obj_i = world.makeRigidObject("object_"+str(i))
    obj_i.geometry().loadFile(params['manipuland_params']['manipuland_fn'])
    obj_i.geometry().transform([scale_x,0,0,0,scale_y,0,0,0,scale_z],[0.0,0.0,0.0])
    obj_i.setTransform(so3.from_quaternion(q_sol[i]),x_sol[i])
    x_plot = [x_sol[i][0]+plot_shift*i,x_sol[i][1],x_sol[i][2]]

    obj_i.setTransform(so3.from_quaternion(q_sol[i]),x_plot)
    obj_i.appearance().setColor(0,0.1*i,1,0.2)
    obj_i.appearance().setSilhouette(0.001)

    # Plot manipulator's contact locations
    point_world = np.zeros(3)
    point_local = np.zeros(3)
    for j in range(len(contact_points)):
        point_world += np.array(se3.apply((so3.from_quaternion(q_sol[i]),x_sol[i]),contact_points[j])) / len(contact_points)
        point_local += np.array(contact_points[j]) / len(contact_points)

    point_local = [point_local[0]*scale_x,point_local[1]*scale_y,point_local[2]*scale_z]

    manipulator_size = 1
    manipulator_i = world.makeRigidObject("manipulator_"+str(i))
    manipulator_i.geometry().loadFile("data/plot/cone.stl")
    manipulator_i.geometry().transform([manipulator_size,0,0,0,manipulator_size,0,0,0,manipulator_size],[0.0,0.0,0.0])

    params['manipuland_params']['manipuland'].setTransform((so3.identity(),[0,0,0]))

    normal_direction = params['manipuland_params']['manipuland'].normal(point_local)
    normal_direction = so3.apply(so3.from_quaternion(q_sol[i]),normal_direction)

    manipulator_i.setTransform(so3.canonical(normal_direction),point_world)
    manipulator_i.appearance().setColor(1,0,0,1.0)
    manipulator_i.appearance().setSilhouette(0.001)
    point_plot = [point_world[0]+plot_shift*i,point_world[1],point_world[2]]
    manipulator_i.setTransform(so3.canonical(normal_direction),point_plot)

vis.add("world",world)
vis.show()

while vis.shown():
    vis.lock()
    t0 = time.time()
    vis.unlock()
    t1 = time.time()
    time.sleep(max(0.001,0.05-(t1-t0)))
vis.kill()