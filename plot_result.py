import json
from klampt import vis
from STOCS import Problem, OptimizerParams, TrajectoryState
from semiinfinite.geometryopt import *
import dacite

import argparse
import importlib
from utils import *
import visualization

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, required=True)

args = parser.parse_args()
result_path = args.result_path

environment,manipuland = result_path.split('/')[-2].split('_')[0],result_path.split('/')[-2].split('_')[1]
module_name = f"cfg.{environment}_{manipuland}"
module = importlib.import_module(module_name)
problem = getattr(module, 'problem', None)   # type: Problem

with open(result_path,'r') as f:
    res = json.load(f)
res = dacite.from_dict(TrajectoryState,res)

import klampt
vis.init()
world = klampt.WorldModel()
vis.setBackgroundColor(1,1,1)

for i,environment in enumerate(problem.environments):
    vis.add("env"+str(i),environment.geom,hide_label=True)
    vis.setColor("env"+str(i),1,0.5,0,1.0)

visualization.plot_trajectory_klampt(res,problem)