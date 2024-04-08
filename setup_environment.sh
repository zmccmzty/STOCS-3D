#!/bin/bash

# Create a virtual environment
python3 -m venv env

# Upgrade pip in the virtual environment
env/bin/pip install --upgrade pip

# Install Drake in the virtual environment
env/bin/pip install drake
env/bin/pip install klampt
env/bin/pip install open3d
env/bin/pip install trimesh
env/bin/pip install functools

# Activate the virtual environment
source env/bin/activate