import numpy as np
from typing import List,Tuple,Optional,Union
from klampt import Geometry3D,VolumeGrid,DistanceQuerySettings,DistanceQueryResult
from klampt.math import se3
from semiinfinite.geometryopt import PenetrationDepthGeometry
import json
from dataclasses import dataclass,field


class SDFCache:
    """Allows evaluating gradient with Drake's autodiff"""
    def __init__(self,geom:Geometry3D):
        assert geom.type() == 'VolumeGrid'
        self.geom = geom
        T0 = geom.getCurrentTransform()
        geom.setCurrentTransform(*se3.identity())
        self.distances = geom.getVolumeGrid()
        self.gradient_x_field = np.empty(self.distances.dims)
        self.gradient_y_field = np.empty(self.distances.dims)
        self.gradient_z_field = np.empty(self.distances.dims)
        bmin = self.distances.bbox[0:3]
        bmax = self.distances.bbox[3:6]
        xgrid = np.linspace(bmin[0],bmax[0],self.distances.dims[0])
        ygrid = np.linspace(bmin[1],bmax[1],self.distances.dims[1])
        zgrid = np.linspace(bmin[2],bmax[2],self.distances.dims[2])
        settings = DistanceQuerySettings()
        for i in range(self.distances.dims[0]):
            for j in range(self.distances.dims[1]):
                for k in range(self.distances.dims[2]):
                    d = self.geom.distance_point_ext((xgrid[i],ygrid[j],zgrid[k]),settings) 
                    grad = d.grad2
                    self.gradient_x_field[i,j,k] = grad[0]
                    self.gradient_y_field[i,j,k] = grad[1]
                    self.gradient_z_field[i,j,k] = grad[2]
        geom.setCurrentTransform(*T0)
    
    def distance(self,pt:Tuple[float,float,float]) -> float:
        return trilinear_interpolation(self.distances.bbox,self.distances.getValues(),pt)
    
    def gradient(self,pt:Tuple[float,float,float]) -> Tuple[float,float,float]:
        dx = trilinear_interpolation(self.distances.bbox,self.gradient_x_field,pt)
        dy = trilinear_interpolation(self.distances.bbox,self.gradient_y_field,pt)
        dz = trilinear_interpolation(self.distances.bbox,self.gradient_z_field,pt)
        return dx,dy,dz

def compute_unified_sdf(geometry_list : List[Union[Geometry3D,PenetrationDepthGeometry]], resolution : Optional[float]=None) -> Geometry3D:
    """Compute a single SDF from a list of geometries.  Result is a klampt
    Geometry3D object with the VolumeGrid datatype.

    If resolution is provided, it's used as the resolution of the SDF.  Otherwise,
    the resolution is chosen to be the smallest of the resolutions of the input.

    The domain of the SDF is determined automatically from the bounding boxes of the
    input geometries.
    """
    if len(geometry_list) == 0:
        raise ValueError("No geometries provided")
    if resolution is None:
        resolution = 0 #flag in Klampt to auto-detect resolution
    g3d_list = []
    for i,g in enumerate(geometry_list):
        if not isinstance(g,(Geometry3D,PenetrationDepthGeometry)):
            raise ValueError(f"Element {i} of geometry_list is not a Geometry3D or PenetrationDepthGeometry")
        if isinstance(g,PenetrationDepthGeometry):
            if g.grid is None:
                raise ValueError(f"PenetrationDepthGeometry must have a grid")
            g3d_list.append(g.grid)
        else:
            g3d_list.append(g)
    if len(g3d_list) == 1:
        if g3d_list[0].type() == 'VolumeGrid':
            return g3d_list[0]   #ignore resolution here
        return g3d_list[0].convert('VolumeGrid',resolution)
    group = Geometry3D()
    group.setGroup()
    for i,g in enumerate(g3d_list):
        group.setElement(i,g)
    res = group.convert('VolumeGrid',resolution)
    return res

    h1, h2, h3 = (x2 - x1) / n1, (y2 - y1) / n2, (z2 - z1) / n3 # grid spacing
    grid_x = np.arange(n1 + 1) * h1 + x1 # grid points in x direction
    grid_y = np.arange(n2 + 1) * h2 + y1 # grid points in y direction
    grid_z = np.arange(n3 + 1) * h3 + z1 # grid points in z direction

    distances = np.zeros((n1 + 1, n2 + 1, n3 + 1)) # distance field
    gradient_x = np.zeros((n1 + 1, n2 + 1, n3 + 1)) # x component of the gradient
    gradient_y = np.zeros((n1 + 1, n2 + 1, n3 + 1)) # y component of the gradient
    gradient_z = np.zeros((n1 + 1, n2 + 1, n3 + 1)) # z component of the gradient

    t_start = time.time()
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            for k, z in enumerate(grid_z):
                if (i * (n1 + 1) * (n2 + 1) + j * (n2 + 1) + k) % 10000 == 0:
                    print(i,j,k)
                    print(f'Computing distance field: {(i * (n2 + 1) * (n3 + 1) + j * (n2 + 1) + k) / ((n1 + 1) * (n2 + 1) * (n3 + 1)) * 100:.2f}%')
                distances[i, j, k] = terrain.distance((x, y, z))[0]

                gradient_x[i, j, k] = (terrain.distance((x + finite_difference_resolution, y, z))[0] - terrain.distance((x - finite_difference_resolution, y, z))[0]) / (2 * finite_difference_resolution)
                gradient_y[i, j, k] = (terrain.distance((x, y + finite_difference_resolution, z))[0] - terrain.distance((x, y - finite_difference_resolution, z))[0]) / (2 * finite_difference_resolution)
                gradient_z[i, j, k] = (terrain.distance((x, y, z + finite_difference_resolution))[0] - terrain.distance((x, y, z - finite_difference_resolution))[0]) / (2 * finite_difference_resolution)

    t_end = time.time()
    print(f'Computing distance field took {t_end - t_start} seconds')

    # create a folder in data to store the following files
    if not os.path.exists('data/environments'):
        os.mkdir('data/environments')
    if not os.path.exists(f'data/environments/{environment_name}'):
        os.mkdir(f'data/environments/{environment_name}')

    params = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'z1': z1, 'z2': z2, 'n1': n1, 'n2': n2, 'n3': n3, 'h1': h1, 'h2': h2, 'h3': h3, 'gridres': gridres, 'pcres': pcres, 'finite_difference_resolution': finite_difference_resolution}

def trilinear_interpolation(bounds, values, pt, is_distance=True):
    """Perform trilinear interpolation on the grid.  Needs to be done here rather than
    klampt.VolumeGrid.distance because drake uses autodiff for gradients.
    """
    bmin = bounds[0:3]
    bmax = bounds[3:6]
    x_bb = (pt[0] - bmin[0]) / (bmax[0] - bmin[0]) * (values.shape[0] - 1)
    y_bb = (pt[1] - bmin[1]) / (bmax[1] - bmin[1]) * (values.shape[1] - 1)
    z_bb = (pt[2] - bmin[2]) / (bmax[2] - bmin[2]) * (values.shape[2] - 1)
    x_idx = int(np.floor(x_bb))
    y_idx = int(np.floor(y_bb))
    z_idx = int(np.floor(z_bb))

    # Check if the point is within the grid
    extra_dist = 0.0
    if x_idx < 0 or x_idx >= values.shape[0] - 1 or y_idx < 0 or y_idx >= values.shape[1] - 1 or z_idx < 0 or z_idx >= values.shape[2] - 1:
        #clamp point to bbox
        closest = np.clip(pt,bmin,bmax)
        if is_distance:
            extra_dist = np.linalg.norm(np.array(pt)-np.array(closest))
        if x_idx < 0:
            x_idx = 0
            x_bb = 0
        if x_idx >= values.shape[0] - 1:
            x_idx = values.shape[0] - 2
            x_bb = values.shape[0] - 1
        if y_idx < 0:
            y_idx = 0
            y_bb = 0
        if y_idx >= values.shape[1] - 1:
            y_idx = values.shape[1] - 2
            y_bb = values.shape[1] - 1
        if z_idx < 0:
            z_idx = 0
            z_bb = 0
        if z_idx >= values.shape[2] - 1:
            z_idx = values.shape[2] - 2
            z_bb = values.shape[2] - 1

    x_u = x_bb - x_idx
    y_u = y_bb - y_idx
    z_u = z_bb - z_idx

    # Perform the interpolations
    c00 = values[x_idx, y_idx, z_idx] * (1 - x_u) + values[x_idx + 1, y_idx, z_idx] * x_u
    c01 = values[x_idx, y_idx, z_idx + 1] * (1 - x_u) + values[x_idx + 1, y_idx, z_idx + 1] * x_u
    c10 = values[x_idx, y_idx + 1, z_idx] * (1 - x_u) + values[x_idx + 1, y_idx + 1, z_idx] * x_u
    c11 = values[x_idx, y_idx + 1, z_idx + 1] * (1 - x_u) + values[x_idx + 1, y_idx + 1, z_idx + 1] * x_u
    
    c0 = c00 * (1 - y_u) + c10 * y_u
    c1 = c01 * (1 - y_u) + c11 * y_u
    
    c = c0 * (1 - z_u) + c1 * z_u
    return c + extra_dist


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


