from __future__ import annotations
import numpy as np
import time
import copy
import math
from klampt import WorldModel
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import SE3Trajectory
from semiinfinite.geometryopt import PenetrationDepthGeometry
import threading
from utils import *
from optimization import Variable,ObjectiveFunction,ConstraintFunction,NonlinearProgramSolver,to_1d_array
from typing import List,Tuple,Dict,Optional,Union,Callable,Any
FloatOrVector = Union[float,np.array]
from klampt.model.typing import RigidTransform,Vector3
from dataclasses import dataclass,replace,field

@dataclass
class Problem:
    """Defines a STOCS trajectory optimization problem."""
    manipuland: PenetrationDepthGeometry
    manipuland_mass : float                 # Mass of the manipuland
    manipuland_com : Vector3                # Center of mass of the manipuland in local frame
    manipuland_inertia: np.ndarray          # Inertia matrix of the manipuland (3x3)
    environments: List[PenetrationDepthGeometry]
    T_init: RigidTransform                  # Initial pose of the manipuland
    T_goal: RigidTransform                  # Goal pose of the manipuland
    manipulation_contact_points: List[Vector3]  # Manipulation contact points, in local frame of manipuland
    manipulation_contact_normals: List[Vector3] # Manipulation contact normals, in local frame of manipuland
    N: int                                  # number of time steps in trajectory
    time_step: float                        # trajectory time step size
    x_bound: Tuple[Vector3,Vector3]         # Position bound
    v_bound: Tuple[Vector3,Vector3]         # Velocity bound
    w_max: float = np.inf                   # Angular velocity bound (magnitude)
    f_max: float = np.inf                   # Overall force bound
    fenv_max: float = np.inf                # Environment contact force bound
    fmnp_max: float = np.inf                # Manipulation contact force bound
    mu_env: float = 1.0                     # Friction coefficient for environment/manipuland contact
    mu_mnp: float = 1.0                     # Friction coefficient for manipulator/manipuland contact
    initial_pose_relaxation: float = 0.0    # Tolerance for initial pose
    goal_pose_relaxation: float = 0.0       # Tolerance for goal pose
    gravity: float = 9.8                    # Gravity constant (acts in -z direction)
    manipuland_name: str = 'manipuland'     # Name of the manipuland, optional
    environment_name: str = 'environment'   # Name of the environment, optional
    environment_sdf_cache: Optional[SDFCache] = None  # Cache for unified environment SDF

    def init_sdf_cache(self, debug=False):
        """Initializes the environment SDF cache.  This is necessary for the
        STOCS optimizer to work.  This function will be called at the start of
        the optimization process.  If debug is True, then the SDF will be
        visualized in the Klamp't visualizer."""
        if self.environment_sdf_cache is None:
            print("Generating environment SDF cache...")
            self.environment_sdf_cache = SDFCache(compute_unified_sdf(self.environments))
            print("Done.")
            if debug:
                from klampt import vis
                environment_mesh = self.environment_sdf_cache.geom.convert('TriangleMesh')
                pc = []
                for v in environment_mesh.getTriangleMesh().getVertices():
                    g = self.environment_sdf_cache.gradient(v)
                    pc.append(vectorops.add(v,vectorops.mul(g,0.05)))
                from klampt.io import numpy_convert
                pc = numpy_convert.from_numpy(np.array(pc),'PointCloud')
                pcg = Geometry3D(pc)
                vis.debug(SDF=environment_mesh,outer_points=pcg,title="Environment SDF")

    @staticmethod
    def from_world(world : WorldModel, rigidObject=0, grid_res=0.005, pc_res=0.005) -> Problem:
        """Constructs the geometric elements of a Problem from a Klamp't WorldModel.

        The manipulator, time steps, bounds, and other parameters must be set manually.
        """
        if world.numRigidObjects() == 0:
            raise ValueError("World has no rigid objects")
        if rigidObject >= world.numRigidObjects():
            raise ValueError("Invalid rigid object index")
        rigidObject = world.rigidObject(rigidObject)
        terrains = [world.terrain(i) for i in range(world.numTerrains())]
        pose = rigidObject.getTransform()
        mass = rigidObject.getMass()
        manipuland_geom = PenetrationDepthGeometry(rigidObject.geometry(),grid_res,pc_res)
        terrain_geoms = [PenetrationDepthGeometry(terrain.geometry(),grid_res,pc_res) for terrain in terrains]
        return Problem(manipuland=manipuland_geom,manipuland_mass=mass.getMass(),manipuland_com=mass.getCom(),manipuland_inertia=mass.getInertia(),
                       environments=terrain_geoms,T_init=pose,T_goal=pose,manipulation_contact_points=[],manipulation_contact_normals=[],N=1,time_step=0.1,
                       x_bound=(vectorops.add(pose[1],[-10]*3),vectorops.add(pose[1],[10]*3)),
                       v_bound = ([-1]*3,[1]*3))
    
    def set_manipulation_from_point(self, point : Vector3, radius_expansion:float=0):
        """Sets the manipulation contact points and normals from a point,
        by assuming the manipuland is at its initial pose, and finding the
        closest point on the manipuland geometry.  All points within this
        radius + radius_expansion will be considered as contact points.
        """
        self.manipuland.setTransform(self.T_init)
        d,p_self,p_other = self.manipuland.distance(point)
        if abs(d) < 1e-6:
            raise ValueError("Point is already in contact with manipuland, can't infer contact normal")
        if radius_expansion == 0:
            self.manipulation_contact_points = [se3.apply(se3.inv(self.T_init,p_self))]
            self.manipulation_contact_normals = [vectorops.unit(vectorops.sub(point,p_self))]
        else:
            self.manipulation_contact_points = []
            self.manipulation_contact_normals = []
            mpts = self.manipuland.pcdata.getPoints()
            point_local = se3.apply(se3.inv(self.T_init,point))
            threshold = (d + radius_expansion)**2
            for i in range(mpts.shape[0]):
                if vectorops.distanceSquared(mpts[i],point_local) <= threshold:
                    self.manipulation_contact_points.append(mpts[i])
                    self.manipulation_contact_normals.append(vectorops.unit(se3.apply(self.T_init,vectorops.sub(point_local,mpts[i]))))


@dataclass
class SmoothingOracleParams:
    """Defines parameters for the spatio-temporal smoothing oracle."""
    add_threshold: float                            # Distance threshold to add active points
    remove_threshold: float                         # Distance threshold to remove active points.  Should be greater than add_threshold
    translation_disturbances: Optional[List[float]]=None   # Disturbances for spatial smoothing
    rotation_disturbances: Optional[List[float]]=None      # Disturbances for spatial smoothing
    time_smoothing_step: int = 1                    # Time step for temporal smoothing. 1 gives reasonable results
    duplicate_detection_threshold: float = 1e-3     # Threshold for detecting duplicate points


@dataclass
class OptimizerParams:
    """Defines parameters for the STOCS optimizer."""
    stocs_max_iter: int                     # maximum number of outer iterations
    oracle_params : SmoothingOracleParams   # Parameters for the spatio-temporal smoothing oracle
    max_mpcc_iter: int = 20                 # maximum number of inner iterations
    initialization: str = 'linear'          # 'linear' or 'initial'
    max_ls_iter : int = 20                  # maximum number of line search iterations
    ls_shrink_factor : float = 0.8          # line search shrink factor
    complementary_convergence_tolerance: float = 1e-4
    equality_convergence_tolerance: float = 1e-4
    inequality_convergence_tolerance: float = 1e-4 
    penetration_convergence_tolerance: float = 1e-4
    step_convergence_tolerance: float = 1e-4
    boundary_convergence_tolerance: float = 1e-4
    major_feasibility_tolerance : float = 1e-4
    major_optimality_tolerance : float = 1e-4
    velocity_complementarity: bool = True   # whether to use velocity complementarity
    comp_threshold_init: float = 1e-4       # Initial complementarity threshold
    comp_threshold_shrink_factor : float =1 # Complementarity threshold shrinking coefficient
    friction_dim: int = 6                   # Friction cone dimension
    assumption: str = 'quasi_dynamic'       # 'quasi_static' or 'quasi_dynamic'
    use_alpha: bool = False                 # Experimental: whether to use alpha in the optimization
    force_on_off: bool = False              # Experimental: ???
    

@dataclass
class TimestepState:
    """Represents a single timestep of a trajectory state in STOCS"""
    x: np.ndarray                           # position
    q: np.ndarray                           # quaternion rotation
    v: np.ndarray                           # velocity
    w: np.ndarray                           # angular velocity
    index_set: List[np.ndarray]             # index set of local points on manipuland in contact
    fenv: List[np.ndarray]                  # environment contact forces
    fmnp: List[np.ndarray]                  # manipulation contact forces
    d: List[float]                          # index set distances
    gamma: List[float]                      # velocity complementarity for each index point (optional)
    dummy: List[float]                      # dummy variable for velocity complementarity (optional)
    alpha: List[float]                      # alpha for each manipulation contact point (optional)

    def pose(self) -> RigidTransform:
        """Returns the manipuland pose as a RigidTransform"""
        return (so3.from_quaternion(self.q),[v for v in self.x])


@dataclass
class TrajectoryState:
    """Main representation of the STOCS solution trajectory and intermediate
    iterates."""
    states: List[TimestepState]              # state for each time step
    times: List[float]                       # time for each time step

    def trajectory(self) -> SE3Trajectory:
        """Returns the manipuland trajectory as an SE3Trajectory"""
        traj = SE3Trajectory()
        for i in range(len(self.states)):
            traj.times.append(self.time_step[i])
            traj.milestones.append(self.states[i].pose())
        return traj


@dataclass
class Result:
    """Defines the result of the STOCS optimizer."""
    is_success: bool
    total_iter: int
    iterates: List[TrajectoryState]
    final: TrajectoryState
    time: float = 0.0

    def average_index_points(self) -> float:
        index_set_nums = [[len(state.index_set) for state in iterate.states] for iterate in self.iterates]
        return np.mean(np.array(index_set_nums))

def collate(vdict:dict,time_index=1):
    """Gathers a dictionary of values whose keys are of the form
    (key,time) into a dictionary of the form key -> [value1,value2,...].
    """
    res = {}
    for k,v in vdict.items():
        if isinstance(k,tuple) and time_index < len(k):
            assert isinstance(k[time_index],int)
            top_key = k[:time_index] + k[time_index+1:]
            t = k[time_index]
            res[top_key] = res.get(top_key,[])
            while len(res[top_key]) < t+1:
                res[top_key].append(None)
            res[top_key][t] = v
        else:
            res[k] = v
    for k,v in res.items():
        if isinstance(v,list):
            try:
                res[k] = np.array(v)
            except Exception:
                pass
    return res

class STOCS(NonlinearProgramSolver):
    """Uses the Simultaneous Trajectory Optimization and Contact Selection
    (STOCS) algorithm to optimize a contact-rich trajectory of a manipuland in
    contact with an environment being manipulated by a point robot.
    """
    def __init__(self, problem : Problem, optimization_params : OptimizerParams):
        super().__init__()
        self.problem = problem
        self.optimization_params = optimization_params
        # Task parameters
        self.N = problem.N
        self.manipuland = self.problem.manipuland
        # current state
        self.current_iterate = None                     # type: Dict[Any,FloatOrVector]
        self.current_index_sets = [[] for i in range(self.N)]   # type: List[List[np.ndarray]]
        #initialize environment SDF if not already initialized
        self.problem.init_sdf_cache()
        # Callback for feedback during optimization
        self.callback = None  # type: Optional[Callable]
        # Complementarity threshold
        self.comp_threshold = None
        # Constant variables and functions
        k = self.optimization_params.friction_dim
        qs = [Variable(('q',i),shape=(4,),description=f'rotation[{i}]') for i in range(self.N)]
        xs = [Variable(('x',i),shape=(3,),lb=np.array(self.problem.x_bound[0]),ub=np.array(self.problem.x_bound[1]),description=f'position[{i}]') for i in range(self.N)]
        qs[0].lb = np.array(so3.quaternion(self.problem.T_init[0]))-self.problem.initial_pose_relaxation
        qs[0].ub = np.array(so3.quaternion(self.problem.T_init[0]))+self.problem.initial_pose_relaxation
        qs[-1].lb = np.array(so3.quaternion(self.problem.T_goal[0]))-self.problem.goal_pose_relaxation
        qs[-1].ub = np.array(so3.quaternion(self.problem.T_goal[0]))+self.problem.goal_pose_relaxation
        xs[0].lb = np.maximum(xs[0].lb,np.array(self.problem.T_init[1])-self.problem.initial_pose_relaxation)
        xs[0].ub = np.minimum(xs[0].ub,np.array(self.problem.T_init[1])+self.problem.initial_pose_relaxation)
        xs[-1].lb = np.maximum(xs[-1].lb,np.array(self.problem.T_goal[1])-self.problem.goal_pose_relaxation)
        xs[-1].ub = np.minimum(xs[-1].ub,np.array(self.problem.T_goal[1])+self.problem.goal_pose_relaxation)
        ws = [Variable(('w',i),shape=(3,),description=f'angular velocity[{i}]') for i in range(self.N)]
        vs = [Variable(('v',i),shape=(3,),lb=np.array(self.problem.v_bound[0]),ub=np.array(self.problem.v_bound[1]),description=f'velocity[{i}]') for i in range(self.N)]
        force_lb = np.zeros(1+k)
        force_ub = np.full(1+k,self.problem.fmnp_max)
        force_ub[1:] *= self.problem.mu_mnp
        m = len(self.problem.manipulation_contact_points)
        f_mnp = [Variable(('f_mnp',i),shape=(m,1+k),lb=np.vstack([force_lb]*m),ub=np.vstack([force_ub]*m),description=f'manipulator force[{i}], friction cone encoding') for i in range(self.N)]
        #should we put f_env in a dynamically changing matrix or individual variables?
        #f_env = [Variable(('f_env',i),shape=(0,1+k),value=np.zeros((0,1+k)),description=f'environment force[{i}], friction cone encoding') for i in range(self.N)]
        for i in range(self.N):
            self.variables[('q',i)] = qs[i]
            self.variables[('x',i)] = xs[i]
            self.variables[('w',i)] = ws[i]
            self.variables[('v',i)] = vs[i]
            self.variables[('f_mnp',i)] = f_mnp[i]
            #self.variables[('f_env',i)] = f_env[i]
        self.objective = ObjectiveFunction(lambda *args:0.0*args[0][0], qs+xs, description='zero objective')
        for i in range(self.N):
            self.constraints[('unit_quaternion',i)] = ConstraintFunction(lambda q:q@q, qs[i],lb=1,ub=1,description=f'unit quaternion[{i}]')
            self.constraints[('angular_velocity_bound',i)] = ConstraintFunction(lambda w:w@w, ws[i],lb=0,ub=self.problem.w_max**2,description=f'angular velocity bound[{i}]')
            self.constraints[('force_balance',i)] = ConstraintFunction(self._force_balance_constraint,[qs[i],xs[i],vs[i],f_mnp[i]],pre_args=(i,),lb=np.zeros(3),ub=np.zeros(3),description=f'force balance[{i}]')
            self.constraints[('torque_balance',i)] = ConstraintFunction(self._torque_balance_constraint,[qs[i],xs[i],ws[i],f_mnp[i]],pre_args=(i,),lb=np.zeros(3),ub=np.zeros(3),description=f'torque balance[{i}]')
            # self.constraints[('force_balance',i)] = ConstraintFunction(self._force_balance_constraint,[qs[i],xs[i],vs[i],f_mnp[i],f_env[i]],pre_args=(i,),lb=np.zeros(3),ub=np.zeros(3),description=f'force balance[{i}]')
            # self.constraints[('torque_balance',i)] = ConstraintFunction(self._torque_balance_constraint,[qs[i],xs[i],ws[i],f_mnp[i],f_env[i]],pre_args=(i,),lb=np.zeros(3),ub=np.zeros(3),description=f'torque balance[{i}]')
            if i != 0:
                self.constraints[('backward_euler_q',i)] = ConstraintFunction(self._backward_euler_q,[qs[i],qs[i-1],ws[i]],rhs=np.zeros(4),description=f'backward euler quaternion[{i}]')
                self.constraints[('backward_euler_x',i)] = ConstraintFunction(self._backward_euler_x,[xs[i],xs[i-1],vs[i]],rhs=np.zeros(3),description=f'backward euler position[{i}]')
            for j in range(len(self.problem.manipulation_contact_points)):
                self.constraints[('fmnp_friction_cone',i,j)] = ConstraintFunction(self._friction_cone_constraint,f_mnp[i][j],pre_args=(self.problem.mu_mnp,),lb=0.0,description=f'f_mnp friction cone[{i},{j}]')

    def set_callback(self, cb : Callable):
        """Sets a callback that will be called every iteration with the current
        iterate before MPCC solving."""
        self.callback = cb

    def oracle(self, iterate : TrajectoryState) -> Tuple[List[List[np.ndarray]],List[List[int]]]:
        """Subclass might override me?

        Returns a tuple (new_index_points,removed_index_points) each of which
        are lists of lists.  The new points are (x,y,z,env_index) tuples, and the
        removed points are indices into the previous index set.
        """
        time_smoothing_step = self.optimization_params.oracle_params.time_smoothing_step

        closest_points = [[] for _ in range(self.N)]
        for i,state in enumerate(iterate.states):           
            closest_points_ti = closest_points[i]

            Ti = state.pose()
            Ri,xi = Ti
            self.manipuland.setTransform(Ti)

            for env_idx, environment in enumerate(self.problem.environments):
                dist_, p_obj, p_env = self.manipuland.distance(environment)
                p_obj_local = se3.apply(se3.inv(Ti),p_obj)

                if dist_< self.optimization_params.oracle_params.add_threshold: 
                    closest_points_ti.append(p_obj_local+[env_idx])

                # Spatial Smoothing
                if self.optimization_params.oracle_params.rotation_disturbances:
                    for disturbance in self.optimization_params.oracle_params.rotation_disturbances:
                        for idx in range(3):
                            if idx == 0:
                                R_p = so3.mul(so3.from_axis_angle(([1,0,0],disturbance)),Ri)
                                R_n = so3.mul(so3.from_axis_angle(([1,0,0],-disturbance)),Ri)
                            elif idx == 1:
                                R_p = so3.mul(so3.from_axis_angle(([0,1,0],disturbance)),Ri)
                                R_n = so3.mul(so3.from_axis_angle(([0,1,0],-disturbance)),Ri)
                            elif idx == 2:
                                R_p = so3.mul(so3.from_axis_angle(([0,0,1],disturbance)),Ri)
                                R_n = so3.mul(so3.from_axis_angle(([0,0,1],-disturbance)),Ri)

                            for R_ in [R_p,R_n]:
                                self.manipuland.setTransform((R_,Ti[1]))
                                dist_, p_obj, p_env = self.manipuland.distance(environment)
                                p_obj_local = se3.apply(se3.inv((R_,Ti[1])),p_obj)

                                if dist_< self.optimization_params.oracle_params.add_threshold:     
                                    closest_points_ti.append(p_obj_local+[env_idx])

                if self.optimization_params.oracle_params.translation_disturbances:
                    for disturbance in self.optimization_params.oracle_params.translation_disturbances:
                        for idx in range(3):
                            if idx == 0:
                                x_p = vectorops.add(xi,[disturbance,0,0])
                                x_n = vectorops.add(xi,[-disturbance,0,0])
                            elif idx == 1:
                                x_p = vectorops.add(xi,[0,disturbance,0])
                                x_n = vectorops.add(xi,[0,-disturbance,0])
                            elif idx == 2:
                                x_p = vectorops.add(xi,[0,0,disturbance])
                                x_n = vectorops.add(xi,[0,0,-disturbance])
                                
                            for x_ in [x_p,x_n]:
                                self.manipuland.setTransform((Ri,x_))
                                dist_, p_obj, p_env = self.manipuland.distance(environment)
                                p_obj_local = se3.apply(se3.inv((Ri,x_)),p_obj)
                                
                                if dist_< self.optimization_params.oracle_params.add_threshold:
                                    closest_points_ti.append(p_obj_local+[env_idx])
        
        # Temporal Smoothing
        index_set = [[] for _ in range(self.N)]
        for ti in range(self.N):
            for t_ in range(ti-time_smoothing_step,ti+time_smoothing_step+1):
                if t_ < 0:
                    pass
                elif t_ > self.N-1:
                    pass
                else: 
                    index_set[ti] += closest_points[t_]

        #do duplicate detection and old point removal
        new_points = [[] for _ in range(self.N)]
        removed_points = [[] for _ in range(self.N)]
        for ti in range(self.N):
            #check if the point is new or existing (TODO: do duplicate checking faster?)
            for point in index_set[ti]:
                dist_min = np.inf
                for point2 in iterate.states[ti].index_set + new_points[ti]:
                    dist = vectorops.norm(vectorops.sub(point,point2))
                    if dist < dist_min:
                        dist_min = dist
                    if dist < self.optimization_params.oracle_params.duplicate_detection_threshold:
                        break
                #new point
                if dist_min > self.optimization_params.oracle_params.duplicate_detection_threshold:
                    new_points[ti].append(point)

            #consider removing points
            for j,point in enumerate(iterate.states[ti].index_set):
                mindist = min(environment.distance(point[:3])[0] for environment in self.problem.environments)
                if mindist > self.optimization_params.oracle_params.remove_threshold:
                    removed_points[ti].append(j)
            
        return new_points,removed_points

    def set_initial_state(self, initial : Optional[Union[TrajectoryState,SE3Trajectory]] = None):
        """Initializes the optimizer according to the initialization parameter
        or with a given initial trajectory.
        """
        if initial is None:
            times = np.arange(0,self.problem.N*self.problem.time_step,self.problem.time_step).tolist()
            fmnp = [np.array([1e-3] + [1e-4]*self.optimization_params.friction_dim) for i in range(len(self.problem.manipulation_contact_points))]
            initial_states = []
            for i in range(self.N):
                T = None
                w = None
                v = None               
                if self.optimization_params.initialization == "initial":
                    T = self.problem.T_init
                    w = np.zeros(3)
                    v = np.zeros(3)
                elif self.optimization_params.initialization == "linear":
                    T = se3.interpolate(self.problem.T_init,self.problem.T_goal,i/(self.N-1))
                    wv = vectorops.mul(se3.error(self.problem.T_goal,self.problem.T_init),1.0/((self.N-1)*self.problem.time_step))
                    w = np.array(wv[0:3])
                    v = np.array(wv[3:6])
                else:
                    raise ValueError(f"Invalid initialization type {self.optimization_params.initialization}")
                
                initial_states.append(TimestepState(x=np.array(T[1]),q=np.array(so3.quaternion(T[0])),v=v,w=w,index_set=[],fenv=[],fmnp=copy.deepcopy(fmnp),d=[],gamma=[],dummy=[],alpha=[]))
            
            self.set_state(TrajectoryState(initial_states, times))
        elif isinstance(initial,TrajectoryState):
            self.set_state(initial)
        else:
            raise NotImplementedError("Setting initial trajectory from SE3Trajectory not yet implemented")
        self.self_check()

    def set_state(self, state : TrajectoryState):
        for i in range(self.N):
            self.current_index_sets[i] = state.states[i].index_set 
            self.variables[('q',i)].set(state.states[i].q)
            self.variables[('x',i)].set(state.states[i].x)
            self.variables[('w',i)].set(state.states[i].w)
            self.variables[('v',i)].set(state.states[i].v)
            self.variables[('f_mnp',i)].set(np.stack(state.states[i].fmnp))
            for j in range(len(state.states[i].index_set)):
                self.variables[('f_env',i,j)].set(state.states[i].fenv[j])
                self.variables[('d',i,j)].set(state.states[i].d[j])
                if self.optimization_params.velocity_complementarity:
                    self.variables[('gamma',i,j)].set(state.states[i].gamma[j])
                    self.variables[('dummy',i,j)].set(state.states[i].dummy[j])
                if self.optimization_params.use_alpha:
                    self.variables[('alpha',i,j)].set(state.states[i].alpha[j])
        self.current_iterate = self.get_var_dict()

    def get_state(self) -> TrajectoryState:
        states = []
        for i in range(self.N):
            m = len(self.problem.manipulation_contact_points)
            k = len(self.current_index_sets[i])
            s = TimestepState(x=self.variables[('x',i)].get(),
                              q=self.variables[('q',i)].get(),
                              v=self.variables[('v',i)].get(),
                              w=self.variables[('w',i)].get(),
                              index_set=self.current_index_sets[i],
                              fenv=[self.variables[('f_env',i,j)].get() for j in range(k)],
                              fmnp=[self.variables[('f_mnp',i)].get()[j] for j in range(m)],
                              d=[self.variables[('d',i,j)].get() for j in range(k)],
                              gamma=[],
                              dummy=[],
                              alpha=[])
        
            if self.optimization_params.velocity_complementarity:
                s.gamma=[self.variables[('gamma',i,j)].get() for j in range(k)]
                s.dummy=[self.variables[('dummy',i,j)].get() for j in range(k)]
            if self.optimization_params.use_alpha:
                s.alpha=[self.variables[('alpha',i,j)].get() for j in range(k)]
            states.append(s)
        times = np.arange(0,self.problem.N*self.problem.time_step,self.problem.time_step).tolist()
        return TrajectoryState(states, times)

    def solve(self) -> Result:
        t_start = time.time()
        iter = 0
        self.comp_threshold = self.optimization_params.comp_threshold_init
        if self.current_iterate is None:
            #initialize, if necessary
            self.set_initial_state()
            #self.pprint_infeasibility()
            
        stocs_res = Result(is_success=False,total_iter=0,iterates=[],final=copy.deepcopy(self.current_iterate))
        print("Initial score function",self._score_function(self.current_iterate))
        if self.callback is not None:
            self.callback(self.current_iterate)
        while iter < self.optimization_params.stocs_max_iter:
            #call oracle
            new_index_set, removed_index_set = self.oracle(self.get_state())

            #update / initialize variables
            for i in range(self.N):
                iset = self.current_index_sets[i]
                #remove index points and variables -- go backwards to avoid index shifting
                for j in removed_index_set[i][::-1]:
                    assert j < len(iset)
                    #self._remove_index_point(i,j)
                
                #add new variables
                self._add_index_points(i,new_index_set[i])
            self.self_check()

            #update trace
            self.current_iterate = self.get_var_dict()
            stocs_res.iterates.append(self.get_state())
            stocs_res.final = stocs_res.iterates[-1]
            stocs_res.total_iter = iter
            print(f"Index points for each time step along the trajectory: {[len(s) for s in self.current_index_sets]}")

            #CONFIGURE THE MPCC SOLVE
            #TODO: THIS IS A HACK! need to make the iteration growth rate a proper parameter
            self.optimization_params.max_mpcc_iter = min(5+5*iter,100)
            #set all the complementarity thresholds
            for i in range(self.N):
                for j in range(len(self.current_index_sets[i])):
                    k = self.optimization_params.friction_dim
                    if ('velocity_complementarity',i,j) in self.constraints:
                        self.constraints[('velocity_complementarity',i,j)].ub[:k] = self.comp_threshold
                    if ('dummy_gamma_complementarity',i,j) in self.constraints:
                        self.constraints[('dummy_gamma_complementarity',i,j)].ub = self.comp_threshold
                    self.constraints[('force_distance_complementarity',i,j)].ub = self.comp_threshold
            print(f"MPCC solve iteration {iter}, current state score {self._score_function(self.current_iterate)}")
            if self.callback is not None:
                self.callback(self.current_iterate)

            def print_dots(stop_event):
                while not stop_event.is_set():
                    print(".", end="", flush=True)
                    time.sleep(1)  

            stop_event = threading.Event()

            dot_thread = threading.Thread(target=print_dots, args=(stop_event,), daemon=True)
            dot_thread.start()
            super().setup_drake_program()
            try:
                #run the NLP solver
                res_target = super().solve(self.optimization_params.max_mpcc_iter,self.optimization_params.major_feasibility_tolerance,self.optimization_params.major_optimality_tolerance,"tmp/debug.txt")
                for k,v in res_target.items():
                    if k[0] == 'q':   #unit quaternions
                        q = res_target[k]
                        res_target[k] = q/np.linalg.norm(q)
                res = self._line_search(res_target,res_current=self.current_iterate)
            finally:
                stop_event.set()
                dot_thread.join()
                print()
            #correct for quaternions
        
            prev_iterate = self.current_iterate
            #update current state
            self.current_iterate = res
            self.set_var_dict(self.current_iterate)

            #convergence check

            #compute state differences
            diffs = []
            compared_vars = ['q','x','w','v']
            for k,v in self.current_iterate.items():
                if k[0] in compared_vars and k in prev_iterate:
                    diffs.append((prev_iterate[k] - v).flatten())
            var_diff = np.concatenate(diffs)
            
            bound_residuals = {}
            complementarity_residuals = {}
            equality_residuals = {}
            inequality_residuals = {}
            for k,v in self.variables.items():
                bound_residuals[k] = v.bound_residual()
            for k,c in self.constraints.items():
                if c.equality():
                    equality_residuals[k] = c.residual()
                elif k[0].endswith('complementarity'):
                    complementarity_residuals[k] = c.residual()
                else:
                    inequality_residuals[k] = c.residual()
            bound_residual = np.concatenate([to_1d_array(r) for r in bound_residuals.values()])
            complementarity_residual = np.concatenate([np.abs(to_1d_array(r)) for r in complementarity_residuals.values()])
            equality_residual = np.concatenate([np.abs(to_1d_array(r)) for r in equality_residuals.values()])
            inequality_residual = np.concatenate([to_1d_array(r) for r in inequality_residuals.values()])
            print("Bound residual",np.max(bound_residual))
            print("Complementarity residual",np.max(complementarity_residual))
            print("Equality residual",np.max(equality_residual))
            print("State delta",np.max(var_diff))
            if np.all(bound_residual < self.optimization_params.boundary_convergence_tolerance) and \
                np.all(complementarity_residual < self.optimization_params.complementary_convergence_tolerance) and \
                np.all(self._deepest_penetration(res) < self.optimization_params.penetration_convergence_tolerance) and \
                np.all(equality_residual < self.optimization_params.equality_convergence_tolerance) and \
                np.all(inequality_residual < self.optimization_params.inequality_convergence_tolerance) and \
                np.linalg.norm(var_diff) < self.optimization_params.step_convergence_tolerance*len(var_diff):
                
                print("Successfully found result.")
                iter += 1
                stocs_res.is_success = True
                break
            
            iter += 1
            self.comp_threshold *= self.optimization_params.comp_threshold_shrink_factor

        stocs_res.iterates.append(self.get_state())
        stocs_res.final = stocs_res.iterates[-1]
        stocs_res.total_iter = iter
        stocs_res.time = time.time()-t_start
        if self.callback is not None:
            self.callback(self.current_iterate)
        return stocs_res
    
    def pprint_infeasibility(self):
        bound_residuals = {}
        complementarity_residuals = {}
        equality_residuals = {}
        inequality_residuals = {}
        for k,v in self.variables.items():
            bound_residuals[k] = v.bound_residual()
        for k,c in self.constraints.items():
            if c.equality():
                equality_residuals[k] = c.residual()
            elif k[0].endswith('complementarity'):
                complementarity_residuals[k] = c.residual()
            else:
                inequality_residuals[k] = c.residual()
        bound_residuals = collate(bound_residuals)
        complementarity_residuals = collate(complementarity_residuals)
        equality_residuals = collate(equality_residuals)
        inequality_residuals = collate(inequality_residuals)
        bound_residuals = dict((k,v) for k,v in bound_residuals.items() if not all(x is None or np.all(x==0) for x in v))
        complementarity_residuals = dict((k,v) for k,v in complementarity_residuals.items() if not all(x is None or np.all(x==0) for x in v))
        equality_residuals = dict((k,v) for k,v in equality_residuals.items() if not all(x is None or np.all(np.abs(x) <= self.optimization_params.equality_convergence_tolerance) for x in v))
        inequality_residuals = dict((k,v) for k,v in inequality_residuals.items() if not all(x is None or np.all(x==0) for x in v))

        if len(bound_residuals) > 0:
            print("Bound residuals:")
            for k,v in bound_residuals.items():
                print(k,v)
        if len(complementarity_residuals) > 0:
            print("Complementarity residuals")
            for k,v in complementarity_residuals.items():
                print(k,v)
        if len(equality_residuals) > 0:
            print("Equality residuals")
            for k,v in equality_residuals.items():
                print(k,v)
        if len(inequality_residuals) > 0:
            print("Inequality residuals")
            for k,v in inequality_residuals.items():
                print(k,v)

    
    def _interpolate(self, iterate1 : dict, iterate2 : dict, u : float, strict=True) -> dict:
        """Interpolates between two states.  If strict is True, then the states'
        lengths must match.  If strict is False, then the states will be
        interpolated even if they have different lengths.
        """
        if strict:
            if len(iterate1) != len(iterate2):
                raise ValueError("Cannot interpolate states with different numbers of variables")
        res = {}
        for k,v1 in iterate1.items():
            v2 = iterate2[k]
            if k[0] == 'q':  #quaternion, do slerp    
                if abs(v1@v1 - 1) > 1e-5:
                    raise ValueError("Start state quaternion is not normalized")
                if abs(v2@v2 - 1) > 1e-5:
                    raise ValueError("End state quaternion is not normalized")
                res[k] = np.array(so3.quaternion(so3.interpolate(so3.from_quaternion(v1),so3.from_quaternion(v2),u)))
            else:
                res[k] = v1 + u*(v2-v1)
        return res

    def _remove_index_point(self, i : int, j : int):
        jlast = len(self.current_index_sets[i])-1
        for var in ['f_env','d','gamma','dummy','alpha']:
            if (var,i,j) not in self.variables:
                continue
            self.variables[(var,i,j)].set(self.variables[(var,i,jlast)].get())
            self.variables[(var,i,j)].solver_impl = self.variables[(var,i,jlast)].solver_impl
            del self.variables[(var,i,jlast)]
        for constraint in ['d_distance_equality','force_distance_complementarity','velocity_complementarity','dummy_friction_residual_constraint','dummy_gamma_complementarity','fenv_friction_cone']:
            if (constraint,i,j) not in self.constraints:
                continue
            self.constraints[(constraint,i,j)] = self.constraints[(constraint,i,jlast)]
            del self.constraints[(constraint,i,jlast)]
        
        fb = self.constraints[('force_balance',i)]
        tb = self.constraints[('torque_balance',i)]
        fb.variables[4+j] = fb.variables.pop()
        tb.variables[4+j] = tb.variables.pop()
        self.current_index_sets[i][j]= self.current_index_sets[i].pop()
        fcount = 0
        for j in range(jlast+1):
            if ('f_env',i,j) in self.variables:
                fcount += 1
        assert fcount == len(self.current_index_sets[i])
        self.self_check()

    def _add_index_points(self,i : int, new_points : List[np.ndarray]):
        m = len(self.current_index_sets[i])
        k = self.optimization_params.friction_dim
        qi = self.variables[('q',i)]
        xi = self.variables[('x',i)]
        vi = self.variables[('v',i)]
        wi = self.variables[('w',i)]
        #initialize force to a small force
        finit = np.array([1e-3] + [1e-4]*k)
        dummy0 = self._friction_cone_constraint(self.problem.mu_env,finit)
        for j in range(m,m+len(new_points)):
            point = new_points[j-m]
            fij = Variable(name=('f_env',i,j),value=finit, lb=np.zeros(1+k),ub=np.array([self.problem.fenv_max]+[self.problem.fenv_max*self.problem.mu_env]*k))
            d0 = self._point_distance(point,qi.get(),xi.get())
            dij = Variable(name=('d',i,j),value=d0,lb=0.0,description=f'environment distance to index point {i}, {j}')

            self.variables[('f_env',i,j)] = fij
            self.variables[('d',i,j)] = dij
            self.constraints[('force_balance',i)].variables.append(fij)
            self.constraints[('torque_balance',i)].variables.append(fij)
            self.constraints[('d_distance_equality',i,j)] = ConstraintFunction(lambda point,q,x,d:self._point_distance(point,q,x)-d,[qi,xi,dij],pre_args=(point,),rhs=0.0)
            def f_d(f,d):
                #print(f,d)
                return f*d
            self.constraints[('force_distance_complementarity',i,j)] = ConstraintFunction(f_d,[fij[0],dij],ub=self.comp_threshold)

            if self.optimization_params.velocity_complementarity:
                g0 = np.linalg.norm(self.variables[('v',i)].get())
                gij = Variable(name=('gamma',i,j),value=g0,lb=0.0,description=f'tangential velocity bound for index point {i}, {j}')
                dummyij = Variable(name=('dummy',i,j),value=dummy0,lb=0.0,description=f'dummy variable for velocity complementarity for index point {i}, {j}')
                self.variables[('gamma',i,j)] = gij
                self.variables[('dummy',i,j)] = dummyij
                def dummy_friction_residual(f,d):
                    return self._friction_cone_constraint(self.problem.mu_env,f)-d
                self.constraints[('dummy_friction_residual_constraint',i,j)] = ConstraintFunction(dummy_friction_residual,[fij,dummyij],rhs=0.0,description=f'Dummy = friction residual [{i},{j}]')
                lb = np.array([-np.inf]*k+[0.]*k)
                ub = np.array([self.comp_threshold]*k+[np.inf]*k)
                self.constraints[('velocity_complementarity',i,j)] = ConstraintFunction(self._velocity_cc_constraint,[qi,wi,xi,vi,gij,fij],pre_args=(point,),lb=lb,ub=ub,description=f"Velocity CC constraint time {i}, index point {j}.") 
                self.constraints[('dummy_gamma_complementarity',i,j)] = ConstraintFunction(lambda d,g: d*g, [dummyij,gij],ub=self.comp_threshold)
            else:
                self.constraints[('fenv_friction_cone',i,j)] = ConstraintFunction(self._friction_cone_constraint,fij,lb=0.0,description=f'fenv friction[{i},{j}]')

                # # Force on and off
                # if self.optimization_params.force_on_off:
                #     if self.optimization_params.use_alpha:
                #         prog.AddConstraint(lambda qxfa:self.force_on_off(self.problem.manipulation_contact_points[j],*np.split(qxfa,[4, 4+3, 7+1])),lb=[4e-2,0],ub=[np.inf,0],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]],alpha_vars[i][j]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")    
                #     else:
                #         prog.AddConstraint(lambda qxf:self.force_on_off(self.problem.manipulation_contact_points[j],np.split(qxf,[4,4+3])),lb=[0.],ub=[self.comp_threshold],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")
                                
            if self.optimization_params.use_alpha:
                self.variables[('alpha',i,j)] = Variable(name='alpha',value=0.0,lb=0.0)

        #add new index points
        self.current_index_sets[i] += new_points

    def evaluate(self,traj : TrajectoryState = None):
        if traj is None:
            super().evaluate()
        else:
            self.set_state(traj)
            super().evaluate()
   
    def _line_search(self, res_target : dict, res_current : dict) -> dict:
        current_score = self._score_function(res_current)
        target_score = self._score_function(res_target)
        res_temp = res_target

        iter_ls = 0
        step_size = 1
        while target_score > current_score and iter_ls < self.optimization_params.max_ls_iter:
            step_size = step_size*self.optimization_params.ls_shrink_factor

            res_temp = self._interpolate(res_current,res_target,step_size) 
            
            target_score = self._score_function(res_temp)

            iter_ls += 1

        return res_temp

    def _score_function(self,res : dict):
        for k,v in self.variables.items():
            v.set(res[k])
        score = self.objective()
        #variable bounds
        for v in self.variables.values():
            score += np.sum(v.bound_residual())
        #constraint residuals
        for k,c in self.constraints.items():
            score += np.sum(np.abs(c.residual()))
        return score

    def _deepest_penetration(self,res: dict) -> list:
        penetration = []
        for i in range(self.N):
            T = so3.from_quaternion(res[('q',i)]),res[('x',i)].tolist()
            self.manipuland.setTransform(T)
            mindist = np.inf
            for environment in self.problem.environments:
                dist_, p_obj, p_env = self.manipuland.distance(environment)
                mindist = min(mindist,dist_)
            penetration.append(-mindist)
        return np.array(penetration)
    
    def _environment_distance_gradient(self,pt_world : Vector3) -> Vector3:
        assert self.problem.environment_sdf_cache is not None, "Environment SDF cache is not initialized."
        return self.problem.environment_sdf_cache.gradient(pt_world)

    def _point_distance(self, point_: Vector3, q, x) -> float:
        assert self.problem.environment_sdf_cache is not None, "Environment SDF cache is not initialized."
        point, env_idx = point_[:3], point_[3]
        point_world = se3.apply((so3.from_quaternion(q),x),point)
        return self.problem.environment_sdf_cache.distance(point_world)
    
    def force_3d(self, f_vec : np.ndarray, normal : Vector3) -> Vector3:
        """Given a force encoded as normal force + tangential forces, returns
        the 3D force vector."""
        k = self.optimization_params.friction_dim
        assert len(f_vec) == 1+k, "Invalid size of f_vec provided to force_3d()"
        N_frame = so3.canonical(normal)
        f = vectorops.mul(normal,f_vec[0])
        for j in range(k):
            phi = (2*math.pi/int(k))*j
            T_friction = so3.apply(N_frame,(0,math.cos(phi),math.sin(phi)))
            f = vectorops.madd(f,T_friction,f_vec[j+1])
        return f
    
    def _force_balance_constraint(self, t : int, q, x, v, f_mnp, *f_env) -> np.ndarray:
        k = self.optimization_params.friction_dim
        assert len(f_env) == len(self.current_index_sets[t])
        for fe in f_env:
            assert len(fe) == k+1
        if len(self.problem.manipulation_contact_points) > 0:
            assert f_mnp.shape == (len(self.problem.manipulation_contact_points),(k+1)), f"Manipulator force dimension is not correct, timestep {t}: {f_mnp.shape} vs {(len(self.problem.manipulation_contact_points),(k+1))}"
        R = so3.from_quaternion(q)

        force = 0
        # Environment contact 
        for i,point in enumerate(self.current_index_sets[t]):
            f_env_i = f_env[i]
            point_world = se3.apply((R,x),point[:3])
            dir_n = np.asarray(self._environment_distance_gradient(point_world))
            N_normal_ = dir_n / np.linalg.norm(dir_n)

            assert np.linalg.norm(dir_n) != 0, "Normal is 0!!!"
            force += np.array(self.force_3d(f_env_i,N_normal_))
        
        # Gravity
        force += np.array([0,0,-self.problem.gravity*self.problem.manipuland_mass])

        # Manipulator contact force
        for i,manipulation_normal in enumerate(self.problem.manipulation_contact_normals):
            f_mnp_i = f_mnp[i]
            force += np.array(so3.apply(R,self.force_3d(f_mnp_i,manipulation_normal)))

        if self.optimization_params.assumption == 'quasi_static':
            return np.array([force[0],force[1],force[2]])
        elif self.optimization_params.assumption == 'quasi_dynamic':
            if t != 0:
                return force-self.problem.manipuland_mass*v/self.problem.time_step
            else:
                return np.array([force[0],force[1],force[2]])

    def _torque_balance_constraint(self,t : int, q, x, w, f_mnp, *f_env) -> np.ndarray:
        k = self.optimization_params.friction_dim
        assert len(f_env) == len(self.current_index_sets[t])
        for fe in f_env:
            assert len(fe) == k+1
        if len(self.problem.manipulation_contact_points) > 0:
            assert f_mnp.shape == (len(self.problem.manipulation_contact_points),(k+1)), f"Manipulator force dimension is not correct, timestep {t}"
        R = so3.from_quaternion(q)
        com_world = se3.apply((R,x),self.problem.manipuland_com)

        torque = 0

        # Contact with the environment
        for i,point in enumerate(self.current_index_sets[t]):
            f_env_i = f_env[i]
            point_world = se3.apply((R,x),point[:3])

            dir_n_ = np.asarray(self._environment_distance_gradient(point_world))
            N_normal = dir_n_ / np.linalg.norm(dir_n_)
            
            force = np.array(self.force_3d(f_env_i,N_normal))
            torque += np.array(vectorops.cross(vectorops.sub(point_world,com_world),force))
        
        # Contact with the manipulator
        for i,(manipulation_contact,manipulation_normal) in enumerate(zip(self.problem.manipulation_contact_points,self.problem.manipulation_contact_normals)):
            f_mnp_i = f_mnp[i]
            force = np.array(so3.apply(R,self.force_3d(f_mnp_i,manipulation_normal))) 
            mnp_contact_world = se3.apply((R,x),manipulation_contact)
            torque += np.array(vectorops.cross(vectorops.sub(mnp_contact_world,com_world),force))

        if self.optimization_params.assumption == 'quasi_static':
            return np.array([torque[0],torque[1],torque[2]])
        elif self.optimization_params.assumption == 'quasi_dynamic':
            I_body = self.problem.manipuland_inertia
            R = np.array(so3.matrix(R))
            I_world = R@I_body@R.T
            if t != 0:
                return torque - I_world@w/self.problem.time_step
            else:
                return np.array([torque[0],torque[1],torque[2]])

    def quat_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=q0.dtype)

    def skew(self,v):

        skew_matrix = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
    
        return skew_matrix

    def _apply_angular_velocity_to_quaternion(self, q, w, dt : float):
        w_mag = w@w
        if w_mag < 1e-15:
            return q
        w_mag = np.sqrt(w_mag)*dt
        #w_axis = w/w_mag
        #delta_q = np.hstack([np.cos(w_mag/2.0), w_axis*np.sin(w_mag/2.0)])
        #sinc (w_mag/2/pi) = sin(pi w_mag/2/pi) / (pi w_mag/2/pi) = 2 sin(w_mag/2)/w_mag
        #w_axis*np.sin(w_mag/2.0) = w/w_mag*np.sin(w_mag/2.0)
        delta_q = np.hstack([np.cos(w_mag*0.5), (0.5)*w*dt*np.sinc(w_mag*0.5/np.pi)])
        return  self.quat_multiply(q, delta_q)

    def _backward_euler_q(self, q, qprev, w):
        return q - self._apply_angular_velocity_to_quaternion(qprev, w, self.problem.time_step)

    def _backward_euler_x(self, x, xprev, v):
        return x - (xprev + v*self.problem.time_step)

    def _force_on_off(self,point, q, x, f, alpha=None):
        raise NotImplementedError("Force_on_off was an experimental hack and is not implemented in the current version of STOCS")
        if self.param.use_alpha:
            point_world = se3.apply((so3.from_quaternion(q),x),point)
            query_object = self.query_object_ad if isinstance(x_[0], AutoDiffXd) else self.query_object_f
            dist = query_object.ComputeSignedDistanceToPoint(p_WQ=point_world)[0].distance

            return np.array([dist+alpha,alpha*f])
        else:
            point_world = se3.apply((so3.from_quaternion(q),x),point)   
            query_object = self.query_object_ad if isinstance(x_[0], AutoDiffXd) else self.query_object_f   
            dist = query_object.ComputeSignedDistanceToPoint(p_WQ=point_world)[0].distance
        
            return np.array([np.log(1+np.exp(-1000*dist))*f])

    def _velocity_cc_constraint(self,point:Vector3,q,w,x,v,gamma,f):
        """Constraint that the velocity opposes frictional forces in 
        tangential directions.

        Returns k*2 elements.  The first k elements are the complementarity
        conditions on tangential velocities and forces, (gamma + v_t[j])*f_t[j] <= 0,
        and the last k elements set conditions on gamma, gamma + v_t[j] >= 0 

        These are coupled with the dummy variable complementarity conditions        

            dummy = fn*mu - sum(f_t[j]) >= 0
            dummy*gamma == 0

        When friction is not at the limit, dummy > 0, and gamma = 0.  This means that both v and gamma
        have to be 0 as desired.
                
        At friction cone limit, we have dummy = 0, which lets gamma be nonzero
        and now v and f must "work together with" gamma a way to satisfy bounds
        given by these constraints.  

        Note that there are options for the complementarity condition.  Method 1
        is (gamma + v_t[j])*f_t[j] <= 0, which
        makes sure that in almost all cases, exactly one friction cone boundary
        edge is active and gamma is the negative velocity along this boundary
        Method 2 is v_t[j]*f_t[j] <= 0, which may lead to less apparent
        friction due to forces being in the interior of the friction cone. 
        TODO: make this a configuration parameter.

        """
        R = so3.from_quaternion(q)
        v_relative = vectorops.cross(w,so3.apply(R,point))
        v_real = vectorops.add(v_relative,v)

        point_world = se3.apply((R,x),point)
        dir_n_ = np.asarray(self._environment_distance_gradient(point_world))
        N_normal = dir_n_ / np.linalg.norm(dir_n_)
        
        v_tangential = vectorops.sub(v_real,vectorops.mul(N_normal,vectorops.dot(v_real,N_normal)))
        N_frame = so3.canonical(N_normal)
        res1 = []
        res2 = []
        for j in range(self.optimization_params.friction_dim):
            phi = (2*math.pi/int(self.optimization_params.friction_dim))*j
            T_friction = so3.apply(N_frame,(0,math.cos(phi),math.sin(phi)))
            # method 1
            res1.append((gamma + vectorops.dot(v_tangential,T_friction))*f[j+1])
            # method 2
            #res1.append(vectorops.dot(v_tangential,T_friction)*f[j+1])
            res2.append(gamma + vectorops.dot(v_tangential,T_friction))
        return np.array(res1+res2)

    def _friction_cone_constraint(self,mu,f_vector):
        fn,ff = np.split(f_vector, [1])
        res = mu*fn[0]
        for ffi in ff:
            res -= ffi
        return res



if __name__ == '__main__':
    # Take arguments from the command line
    import argparse
    import importlib
    import time
    import os
    from klampt import vis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=str, default='plane', help='Environment')
    parser.add_argument('--manipuland', type=str, default='mustard', help='Manipuland')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()
    environment = args.environment
    manipuland = args.manipuland

    # Construct the module name based on the command line arguments
    module_name = f"cfg.{environment}_{manipuland}"
    print(f"Environment: {environment}")
    print(f"Manipuland: {manipuland}")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    #importlib.reload(module)
    # Access the 'problem' and 'params' attribute from the imported module
    problem = getattr(module, 'problem', None)
    params = getattr(module, 'params', None)

    if problem is None:
        print(f"The 'problem' attribute was not found in the module {module_name}")
        exit(1)
    elif params is None:
        print(f"The 'params' attribute was not found in the module {module_name}")
        exit(1)
    else:
        print(f"Imported 'problem' and 'params' from {module_name}")
        print(f"Environment: {environment}")
        print(f"Manipuland: {manipuland}")
        print(f"Number of time steps: {problem.N}")
        print(f"Time step: {problem.time_step}")
        if args.debug:
            print(f"Debug mode is on")
            problem.init_sdf_cache(debug=True)
            from visualization import plot_problem_klampt,plot_trajectory_klampt
            #plot_problem_klampt(problem,show=False)
            #vis.show()
            #while vis.shown():
            #    time.sleep(0.1)
            #vis.show(False)

    problem.manipuland_name = manipuland
    problem.environment_name = environment
    stocs_3d = STOCS(problem,params)

    if args.debug:
        def vis_update(iterate):
            print("Performed vis_update")
            plot_trajectory_klampt(iterate,problem,False)
            time.sleep(0.1)
        stocs_3d.set_callback(vis_update)
        vis.show()
        plot_problem_klampt(problem,False)

    import warnings
    warnings.filterwarnings("error")
    try:
       res = stocs_3d.solve()
    except Warning as e:
        raise()
    print(f"Time used: {res.time}s.")
    print(f"Success: {res.is_success}")
    print(f"Outer iterations: {res.total_iter}")
    print(f"Average index point: {res.average_index_points()}")

    # Save the result as JSON
    import dataclasses
    import json
    folder = f'results/{environment}_{manipuland}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("Saving results to", folder)
    with open(f'{folder}/result.json', 'w') as f:
        json.dump(dataclasses.asdict(res),f,cls=NumpyEncoder)
    with open(f'{folder}/params.json', 'w') as f:
        json.dump(dataclasses.asdict(params),f,cls=NumpyEncoder)
    with open(f'{folder}/problem.json', 'w') as f:
        replaced_problem = replace(problem,manipuland=None,environments=[],environment_sdf_cache=None)
        json.dump(dataclasses.asdict(replaced_problem),f,cls=NumpyEncoder)
    
    if args.debug:
        while vis.shown():
            time.sleep(0.1)
        vis.kill()
