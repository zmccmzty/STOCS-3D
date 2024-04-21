from __future__ import annotations
import numpy as np
import time
import copy
import math
from functools import partial
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import SE3Trajectory
from pydrake.all import MathematicalProgram, Variable, Solve, eq, le, ge, SolverOptions, SnoptSolver, IpoptSolver, NloptSolver, CommonSolverOption
from pydrake.autodiffutils import AutoDiffXd
from semiinfinite.geometryopt import *
import threading
from utils import *
from typing import List,Tuple,Dict,Optional,Union,Callable
from klampt.model.typing import RigidTransform,Vector3
from dataclasses import dataclass,replace

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
                vis.debug(SDF=environment_mesh,title="Environment SDF")

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
    balance_convergence_tolerance: float = 1e-4
    penetration_convergence_tolerance: float = 1e-4
    step_convergence_tolerance: float = 1e-4
    boundary_convergence_tolerance: float = 1e-4
    major_feasibility_tolerance : float = 1e-4
    major_optimality_tolerance : float = 1e-4
    velocity_complementarity: bool = True   # whether to use velocity complementarity
    comp_threshold: float = 1e-4            # Complementarity threshold
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
    gamma: List[float]                      # velocity complementarity for each index point
    dummy: List[float]                      # dummy variable for velocity complementarity
    alpha: List[float]                      # alpha for each manipulation contact point

    def pose(self) -> RigidTransform:
        """Returns the manipuland pose as a RigidTransform"""
        return (so3.from_quaternion(self.q),[v for v in self.x])

    @staticmethod
    def interpolate(a : TimestepState, b : TimestepState, u : float, strict=True) -> TimestepState:
        """Interpolates between two states.  If strict is True, then the states'
        lengths must match.  If strict is False, then the states will be
        interpolated even if they have different lengths.
        """
        if strict:
            if len(a.index_set) != len(b.index_set):
                raise ValueError("Cannot interpolate states with different index sets")
            if len(a.fenv) != len(b.fenv):
                raise ValueError("Cannot interpolate states with different environment force lengths")
            if len(a.fmnp) != len(b.fmnp):
                raise ValueError("Cannot interpolate states with different manipulation force lengths")
            if len(a.d) != len(b.d):
                raise ValueError("Cannot interpolate states with different distance lengths")
            if len(a.gamma) != len(b.gamma):
                raise ValueError("Cannot interpolate states with different gamma lengths")
            if len(a.dummy) != len(b.dummy):
                raise ValueError("Cannot interpolate states with different dummy lengths")
            if len(a.alpha) != len(b.alpha):
                raise ValueError("Cannot interpolate states with different alpha lengths")
        x = a.x + u*(b.x-a.x)
        if abs(a.q@a.q - 1) > 1e-5:
            raise ValueError("Start state quaternion is not normalized")
        if abs(b.q@b.q - 1) > 1e-5:
            raise ValueError("End state quaternion is not normalized")
        q = so3.quaternion(so3.interpolate(so3.from_quaternion(a.q),so3.from_quaternion(b.q),u))
        w = a.w + u*(b.w-a.w)
        v = a.v + u*(b.v-a.v)
        index_set = [vectorops.interpolate(a.index_set[i],b.index_set[i],u) for i in range(len(a.index_set))]
        fenv = [a.fenv[i]+u*(b.fenv[i]-a.fenv[i]) for i in range(len(a.fenv))]
        fmnp = [a.fmnp[i]+u*(b.fmnp[i]-a.fmnp[i]) for i in range(len(a.fmnp))]
        d = vectorops.interpolate(a.d,b.d,u)
        gamma = vectorops.interpolate(a.gamma,b.gamma,u)
        dummy = vectorops.interpolate(a.dummy,b.dummy,u)
        alpha = vectorops.interpolate(a.alpha,b.alpha,u)
        return TimestepState(x=x,q=np.array(q),v=v,w=w,index_set=index_set,fenv=fenv,fmnp=fmnp,d=d,gamma=gamma,dummy=dummy,alpha=alpha)


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

    def flatten(self) -> Dict[str,np.ndarray]:
        """Creates a dictionary of flattened state variables as matrices, tensors,
        vectors (if scalar or variable sized)"""
        q = [s.q for s in self.states]
        x = [s.x for s in self.states]
        w = [s.w for s in self.states]
        v = [s.v for s in self.states]
        fenv = [np.concatenate(s.fenv) for s in self.states]
        fmnp = [np.concatenate(s.fmnp) for s in self.states]
        return {'q':np.array(q),'x':np.array(x),'w':np.array(w),'v':np.array(v),'fenv':np.concatenate(fenv),'fmnp':np.concatenate(fmnp)}

    @staticmethod
    def interpolate(a : TrajectoryState, b: TrajectoryState, u : float) -> TrajectoryState:
        """Interpolates between two iterates"""
        assert a.times == b.times
        return TrajectoryState([TimestepState.interpolate(a.states[i],b.states[i],u) for i in range(len(a.states))],a.times)


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


class STOCS(object):
    """Uses the Simultaneous Trajectory Optimization and Contact Selection
    (STOCS) algorithm to optimize a contact-rich trajectory of a manipuland in
    contact with an environment being manipulated by a point robot.
    """
    def __init__(self, problem : Problem, optimization_params : OptimizerParams):
        self.problem = problem
        self.optimization_params = optimization_params
        # Task parameters
        self.N = problem.N
        self.manipuland = self.problem.manipuland
        # current state
        self.current_iterate = None
        #initialize environment SDF if not already initialized
        self.problem.init_sdf_cache()

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
            fmnp = [np.array([1e-3] + [1e-3]*self.optimization_params.friction_dim) for i in range(len(self.problem.manipulation_contact_points))]
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
                    wv = vectorops.mul(se3.error(self.problem.T_init,self.problem.T_goal),1.0/self.problem.time_step)
                    w = np.array(wv[0:3])
                    v = np.array(wv[3:6])
                else:
                    raise ValueError(f"Invalid initialization type {self.optimization_params.initialization}")
                
                initial_states.append(TimestepState(x=np.array(T[1]),q=np.array(so3.quaternion(T[0])),v=v,w=w,index_set=[],fenv=[],fmnp=copy.deepcopy(fmnp),d=[],gamma=[],dummy=[],alpha=[]))
            self.current_iterate = TrajectoryState(initial_states, times)
        elif isinstance(initial,TrajectoryState):
            self.current_iterate = TrajectoryState
        else:
            raise NotImplementedError("Setting initial trajectory from SE3Trajectory not yet implemented")

    def solve(self) -> Result:
        t_start = time.time()
        iter = 0
        if self.current_iterate is None:
            #initialize, if necessary
            self.set_initial_state()

        stocs_res = Result(is_success=False,total_iter=0,iterates=[],final=replace(self.current_iterate))
        mpcc = MPCC(self.problem,self.optimization_params,self.current_iterate)
        print("Initial score function",self._score_function(mpcc,self.current_iterate))
        while iter < self.optimization_params.stocs_max_iter:
            #call oracle
            new_index_set, removed_index_set = self.oracle(self.current_iterate)

            #update / initialize variables
            for i in range(self.N):
                statei = self.current_iterate.states[i]
                #remove index points and variables -- go backwards to avoid index shifting
                for j in removed_index_set[i][::-1]:
                    assert j < len(statei.index_set)
                    statei.index_set.pop(j)
                    statei.fenv.pop(j)
                    statei.d.pop(j)
                    if self.optimization_params.velocity_complementarity:
                        statei.gamma.pop(j)
                        statei.dummy.pop(j)
                    if self.optimization_params.use_alpha:
                        statei.alpha.pop(j)
                
                #add new variables
                if self.optimization_params.velocity_complementarity:
                    statei.gamma += [0.0]*len(new_index_set[i])
                    statei.dummy += [0.0]*len(new_index_set[i])

                for j in range(len(new_index_set[i])):
                    #initialize force to a small force
                    statei.fenv.append(np.array([1e-3] + [1e-3]*self.optimization_params.friction_dim))
                
                statei.d += [0.0]*len(new_index_set[i])
                
                if self.optimization_params.use_alpha:
                    statei.alpha += [0.0]*len(new_index_set[i])

                #add new index points
                statei.index_set += new_index_set[i]

            #update trace
            stocs_res.iterates.append(copy.deepcopy(self.current_iterate))
            stocs_res.final = stocs_res.iterates[-1]
            stocs_res.total_iter = iter
            print(f"Index points for each time step along the trajectory: {[len(s.index_set) for s in self.current_iterate.states]}")

            #TODO: THIS IS A HACK! need to make the iteration growth rate a proper parameter
            self.optimization_params.max_mpcc_iter = min(5+5*iter,100)
            mpcc = MPCC(self.problem,self.optimization_params,self.current_iterate)
            print(f"MPCC solve iteration {iter}, current state score {self._score_function(mpcc,self.current_iterate)}")

            def print_dots(stop_event):
                while not stop_event.is_set():
                    print(".", end="", flush=True)
                    time.sleep(1)  

            stop_event = threading.Event()

            dot_thread = threading.Thread(target=print_dots, args=(stop_event,), daemon=True)
            dot_thread.start()
            try:
                res_target = mpcc.solve()
                res = self._line_search(mpcc,res_target,res_current=self.current_iterate)
            finally:
                stop_event.set()
                dot_thread.join()
                print()
        
            prev_flat = self.current_iterate.flatten()
            new_flat = res.flatten()
            self.current_iterate = res

            #convergence check

            #compute state differences
            diffs = [(prev_flat[s_var] - new_flat[s_var]).flatten() for s_var in ['q','x','w','v']]
            var_diff = np.concatenate(diffs)
            
            n_index_points = sum(len(s.index_set) for s in self.current_iterate.states)
            if self._complimentarity_residual(mpcc,res) < self.optimization_params.complementary_convergence_tolerance*(1+1+self.optimization_params.friction_dim)*n_index_points and \
                self._deepest_penetration(res) < self.optimization_params.penetration_convergence_tolerance*self.N and \
                self._balance_residual(mpcc,res) < self.optimization_params.balance_convergence_tolerance*self.N and \
                np.linalg.norm(var_diff) < self.optimization_params.step_convergence_tolerance*len(var_diff):
                
                print("Successfully found result.")
                iter += 1
                stocs_res.is_success = True
                break
            
            iter += 1

        stocs_res.time = time.time()-t_start
        return stocs_res

    def _score_function(self,mpcc : MPCC,res : TrajectoryState):
        qtraj = [s.q for s in res.states]
        xtraj = [s.x for s in res.states]
        score = mpcc.cost_fun(qtraj,xtraj) + self._constraint_violation(mpcc,res)
        return score

    def _line_search(self, mpcc : MPCC,res_target : TrajectoryState, res_current : TrajectoryState):
        current_score = self._score_function(mpcc,res_current)
        target_score = self._score_function(mpcc,res_target)
        res_temp = res_target

        iter_ls = 0
        step_size = 1
        while target_score > current_score and iter_ls < self.optimization_params.max_ls_iter:
            step_size = step_size*self.optimization_params.ls_shrink_factor

            res_temp = TrajectoryState.interpolate(res_current,res_target,step_size) 
            
            target_score = self._score_function(mpcc,res_temp)

            iter_ls += 1

        return res_temp

    def _constraint_violation(self,mpcc : MPCC, res: TrajectoryState):
        score = self._index_point_penetration(res) + self._deepest_penetration(res) + \
                self._complimentarity_residual(mpcc,res) + self._balance_residual(mpcc,res) +\
                self._unit_constraint_violation(res) + self._boundary_constraint_violation(res) +\
                self._integration_constraint_violation(mpcc,res)
        return score

    def _unit_constraint_violation(self, res : TrajectoryState):
        violation = 0
        for i in range(self.N):
            q = res.states[i].q
            violation += np.abs(q@q-1)
        
        return violation

    def _boundary_constraint_violation(self, res : TrajectoryState):
        violation = 0
        for i in range(self.N):
            xvio = np.maximum(np.maximum(res.states[i].x-self.problem.x_bound[1],self.problem.x_bound[0]-res.states[i].x),0.0)
            vvio = np.maximum(np.maximum(res.states[i].v-self.problem.v_bound[1],self.problem.v_bound[0]-res.states[i].v),0.0)
            violation += np.sum(xvio)
            violation += np.sum(vvio)

            violation += max(0,vectorops.norm(res.states[i].w)-self.problem.w_max)


            for j in range(len(res.states[i].index_set)):
                for k in range(self.optimization_params.friction_dim+1):
                    violation += max(0,-res.states[i].fenv[j][k])
                violation += max(0,res.states[i].fenv[j][0]-self.problem.fenv_max)

                violation += max(0,-res.states[i].d[j])

            for j in range(len(self.problem.manipulation_contact_points)):
                for k in range(self.optimization_params.friction_dim+1):
                    violation += max(0,-res.states[i].fmnp[j][k])
                violation += max(0,res.states[i].fmnp[j][0]-self.problem.fmnp_max)

            if self.optimization_params.velocity_complementarity:
                for j in range(len(res.states[i].index_set)):
                    violation += max(0,-res.states[i].gamma[j])
                    violation += max(0,-res.states[i].dummy[j])

        return violation

    def _integration_constraint_violation(self,mpcc : MPCC,res : TrajectoryState):
        violation = 0
        for i in range(1,self.N):
            qerr = mpcc.backward_euler_q(res.states[i].q,res.states[i-1].q,res.states[i].w,self.problem.time_step)
            xerr = mpcc.backward_euler_x(res.states[i].x,res.states[i-1].x,res.states[i].v,self.problem.time_step)
            violation += np.sum(np.abs(qerr)) + np.sum(np.abs(xerr))

        return violation

    def _index_point_penetration(self,res : TrajectoryState):
        penetration = 0
        for i in range(self.N):
            T = res.states[i].pose()
            for point in res.states[i].index_set:
                point_world = se3.apply(T,point)   
                for environment in self.problem.environments:
                    dist = environment.distance(point_world)[0]
                    penetration += abs(min(dist,0))
        return penetration

    def _deepest_penetration(self,res: TrajectoryState):
        penetration = 0
        for i in range(self.N):
            T = res.states[i].pose()
            self.manipuland.setTransform(T)
            for environment in self.problem.environments:
                dist_, p_obj, p_env = self.manipuland.distance(environment)
                if dist_ < 0:
                    penetration += abs(min(dist_,0))

        return penetration

    def _complimentarity_residual(self,mpcc : MPCC, res : TrajectoryState):
        residual = 0
        for i in range(self.N):
            s = res.states[i]
            for j in range(len(s.index_set)):
                # Position Comp
                residual += abs(min(self.optimization_params.comp_threshold-s.fenv[j][0]*s.d[j],0))
                
                # Velocity Comp
                if self.optimization_params.velocity_complementarity:
                    constraint_value = mpcc.velocity_cc_constraint(res.states[i].index_set[j],s.q,s.x,s.v,s.w,s.gamma[j],s.fenv[j])
                    residual += np.sum(np.abs(np.minimum(self.optimization_params.comp_threshold - constraint_value[0:self.optimization_params.friction_dim],0)))               
                    residual += abs(min(self.optimization_params.comp_threshold - s.dummy[j]*s.gamma[j],0))

        return residual

    def _balance_residual(self,mpcc : MPCC, res : TrajectoryState):
        residual = 0
        for i in range(self.N):
            s = res.states[i]
            residual += np.sum(np.abs(mpcc.force_balance_constraint(i,s.q,s.x,s.v,np.asarray(s.fenv),np.asarray(s.fmnp))))
            residual += np.sum(np.abs(mpcc.torque_balance_constraint(i,s.q,s.x,s.w,np.asarray(s.fenv),np.asarray(s.fmnp))))

        return residual


class MPCC(object):
    def __init__(self,problem: Problem, param : OptimizerParams, iterate: TrajectoryState):

        self.problem = problem
        self.param = param
        self.iterate = iterate

        # Optimization parameters
        self.N = self.problem.N 
        self.friction_dim = self.param.friction_dim
        self.prog = None

        assert problem.environment_sdf_cache is not None, "Environment SDF cache is not initialized."

    def environment_distance(self,pt_world):
        return self.problem.environment_sdf_cache.distance(pt_world)

    def environment_distance_gradient(self,pt_world):
        return self.problem.environment_sdf_cache.gradient(pt_world)

    def point_distance_constraint(self,point_, q, x, d):
        point, env_idx = point_[:3], point_[3]

        point_world = se3.apply((so3.from_quaternion(q),x),point)     
        dist = self.environment_distance(point_world)

        res = dist - d

        return np.array([res])
    
    def force_3d(self, f_vec : np.ndarray, normal : Vector3) -> Vector3:
        """Given a force encoded as normal force + tangential forces, returns
        the 3D force vector."""
        assert len(f_vec) == 1+self.friction_dim, "Invalid size of f_vec provided to force_3d()"
        N_frame = so3.canonical(normal)
        f = vectorops.mul(normal,f_vec[0])
        for j in range(self.friction_dim):
            phi = (2*math.pi/int(self.friction_dim))*j
            T_friction = so3.apply(N_frame,(0,math.cos(phi),math.sin(phi)))
            f = vectorops.madd(f,T_friction,f_vec[j+1])
        return f
    
    def force_balance_constraint(self, t : int, q, x, v, f_env, f_mnp):
        if len(self.iterate.states[t].index_set) > 0:
            assert f_env.shape == (len(self.iterate.states[t].index_set),(self.friction_dim+1)), f"Environment force dimension is not correct, timestep {t}: {f_env.shape} vs {(len(self.iterate.states[t].index_set),(self.friction_dim+1))}"
        if len(self.problem.manipulation_contact_points) > 0:
            assert f_mnp.shape == (len(self.problem.manipulation_contact_points),(self.friction_dim+1)), f"Manipulator force dimension is not correct, timestep {t}: {f_mnp.shape} vs {(len(self.problem.manipulation_contact_points),(self.friction_dim+1))}"
        R = so3.from_quaternion(q)

        force = 0
        # Environment contact 
        for i,point in enumerate(self.iterate.states[t].index_set):
            f_env_i = f_env[i]
            point_world = se3.apply((R,x),point[:3])
            dir_n = np.asarray(self.environment_distance_gradient(point_world))
            N_normal_ = dir_n / np.linalg.norm(dir_n)

            assert np.linalg.norm(dir_n) != 0, "Normal is 0!!!"
            force += np.array(self.force_3d(f_env_i,N_normal_))
        
        # Gravity
        force += np.array([0,0,-self.problem.gravity*self.problem.manipuland_mass])

        # Manipulator contact force
        for i,manipulation_normal in enumerate(self.problem.manipulation_contact_normals):
            f_mnp_i = f_mnp[i]
            force += np.array(so3.apply(R,self.force_3d(f_mnp_i,manipulation_normal)))

        if self.param.assumption == 'quasi_static':
            return np.array([force[0],force[1],force[2]])
        elif self.param.assumption == 'quasi_dynamic':
            if t != 0:
                return force-self.problem.manipuland_mass*v/self.problem.time_step
            else:
                return np.array([force[0],force[1],force[2]])

    def torque_balance_constraint(self,t : int, q, x, w, f_env, f_mnp):
        if len(self.iterate.states[t].index_set) > 0:
            assert f_env.shape == (len(self.iterate.states[t].index_set),(self.friction_dim+1)), f"Environment force dimension is not correct, timestep {t}"
        if len(self.problem.manipulation_contact_points) > 0:
            assert f_mnp.shape == (len(self.problem.manipulation_contact_points),(self.friction_dim+1)), f"Manipulator force dimension is not correct, timestep {t}"
        R = so3.from_quaternion(q)
        com_world = se3.apply((R,x),self.problem.manipuland_com)

        torque = 0

        # Contact with the environment
        for i,point in enumerate(self.iterate.states[t].index_set):
            f_env_i = f_env[i]
            point_world = se3.apply((R,x),point[:3])

            dir_n_ = np.asarray(self.environment_distance_gradient(point_world))
            N_normal = dir_n_ / np.linalg.norm(dir_n_)
            
            force = np.array(self.force_3d(f_env_i,N_normal))
            torque += np.array(vectorops.cross(vectorops.sub(point_world,com_world),force))
        
        # Contact with the manipulator
        for i,(manipulation_contact,manipulation_normal) in enumerate(zip(self.problem.manipulation_contact_points,self.problem.manipulation_contact_normals)):
            f_mnp_i = f_mnp[i]
            force = np.array(so3.apply(R,self.force_3d(f_mnp_i,manipulation_normal))) 
            mnp_contact_world = se3.apply((R,x),manipulation_contact)
            torque += np.array(vectorops.cross(vectorops.sub(mnp_contact_world,com_world),force))

        if self.param.assumption == 'quasi_static':
            return np.array([torque[0],torque[1],torque[2]])
        elif self.param.assumption == 'quasi_dynamic':
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

    def apply_angular_velocity_to_quaternion(self, q, w, dt):
        w_mag = np.linalg.norm(w)*dt
        #w_axis = w/w_mag
        #delta_q = np.hstack([np.cos(w_mag/2.0), w_axis*np.sin(w_mag/2.0)])
        #sinc (w_mag/2) = sin(pi w_mag/2) / (pi w_mag/2)
        delta_q = np.hstack([np.cos(w_mag*0.5), (np.pi*0.5)*w*np.sinc(w_mag*0.5)])
        return  self.quat_multiply(q, delta_q)

    def backward_euler_q(self, q, qprev, w, dt):
        return q - self.apply_angular_velocity_to_quaternion(qprev, w, dt)

    def backward_euler_x(self, x, xprev, v, dt):
        return x - (xprev + v*dt)

    def force_on_off(self,point, q, x, f, alpha=None):
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

    def velocity_cc_constraint(self,point:Vector3,q,w,x,v,gamma,f):
        """Constraint that the velocity opposes frictional forces in 
        tangential directions.

        Returns k*2 elements.  The first k elements are the complementarity
        conditions on tangential velocities and forces, v_t[j]*f_t[j] <= 0,
        and the last k elements set conditions on gamma, gamma + v_t[j] >= 0 

        These are coupled with the dummy variable complementarity conditions        

            dummy = fn*mu - sum(f_t[j]) >= 0
            dummy*gamma == 0

        When friction is not at the limit, dummy > 0, and gamma = 0.  This means that both v and gamma
        have to be 0 as desired.
                
        At friction cone limit, we have dummy = 0, which lets v and f be anything that
        gives gamma a way to satisfy bounds given by these constraints.

        """
        R = so3.from_quaternion(q)
        v_relative = vectorops.cross(w,so3.apply(R,point))
        v_real = vectorops.add(v_relative,v)

        point_world = se3.apply((R,x),point)
        dir_n_ = np.asarray(self.environment_distance_gradient(point_world))
        N_normal = dir_n_ / np.linalg.norm(dir_n_)
        
        v_tangential = vectorops.sub(v_real,vectorops.mul(N_normal,vectorops.dot(v_real,N_normal)))
        N_frame = so3.canonical(N_normal)
        res1 = []
        res2 = []
        for j in range(self.friction_dim):
            phi = (2*math.pi/int(self.friction_dim))*j
            T_friction = so3.apply(N_frame,(0,math.cos(phi),math.sin(phi)))
            #res1.append(gamma + vectorops.dot(v_tangential,T_friction)*f[j+1])
            res1.append(vectorops.dot(v_tangential,T_friction)*f[j+1])
            res2.append(gamma + vectorops.dot(v_tangential,T_friction))
        return np.array(res1+res2)

    def friction_cone_constraint(self,mu,f_vector):
        fn,ff = np.split(f_vector, [1])
        res = mu*fn 
        for ffi in ff:
            res -= ffi
        return np.array([res])

    def dummy_friction_residual_constraint(self,mu,f_vector,residual):
        fn,ff = np.split(f_vector, [1])
        res = mu*fn
        for ffi in ff:
            res -= ffi
        res -= residual
        return np.array([res])    

    def cost_fun(self,q_traj,x_traj):
        return 0.0*x_traj[0][0]
    
    def wrap(self,func:Callable,vars:list,
             pre_args:Tuple=(),
             post_args:Tuple=(),
             kwargs:Optional[Dict]=None) -> Tuple[Callable,np.ndarray]:
        """Automatically encodes and decodes heterogeneous variables to a
        function into a Drake function and list of variables."""
        if pre_args or post_args or kwargs:
            if kwargs is None:
                kwargs = {}
            def new_func(*args,pre=pre_args,post=post_args,kw=kwargs):
                return func(*(pre + args + post),**kwargs)
            return self.wrap(new_func,vars)

        all_scalar = True
        concat_list = []
        size_list = []
        for v in vars:
            if isinstance(v,Variable):
                concat_list.append([v])
                size_list.append(1)
            else:
                if len(v.shape) == 1:
                    concat_list.append(v)
                    size_list.append(len(v))
                elif len(v.shape) == 2:
                    concat_list.append(v.flatten())
                    size_list.append(v.shape[0]*v.shape[1])
                else:
                    raise NotImplementedError("Only scalar, 1D, and 2D variables are supported")
                all_scalar = False
        if all_scalar:
            # just pass through
            return func,vars
        else:
            split_list = np.cumsum(size_list)[:-1]
            def flattened_func(x):
                x_split = np.split(x,split_list)
                for i in range(len(x_split)):
                    if isinstance(vars[i],Variable):
                        x_split[i] = x_split[i][0]
                    elif len(vars[i].shape)==2:
                        x_split[i] = x_split[i].reshape(vars[i].shape)
                return func(*x_split)
            return flattened_func,np.concatenate(concat_list)

    def add_constraint(self,func:Callable,lb:np.ndarray,ub:np.ndarray,vars:list,
                       description:Optional[str]=None,
                       pre_args:Tuple=(),
                       post_args:Tuple=(),
                       kwargs:Optional[Dict]=None):
        """Automatically encodes and decodes heterogeneous variables to a
        function into a Drake constraint.
        
        The function is assumed to be called in the form `func(v1,v2,...,vn)`,
        unless pre_args is given, in which case it is assumed to be called
        `func(pre_args_1,...,pre_args_m,v1,v2,...,vn)`.

        This will handle all of the stacking and unstacking of the variables.
        """
        flat_func,flat_vars = self.wrap(func,vars,pre_args,post_args,kwargs)
        c = self.prog.AddConstraint(flat_func,lb,ub,flat_vars)
        if description:
            c.evaluator().set_description(description)
    
    def add_cost(self,func:Callable,vars:list,
                    description:Optional[str]=None,
                    pre_args:Tuple=(),
                    post_args:Tuple=(),
                    kwargs:Optional[Dict]=None):
        """Automatically encodes and decodes heterogeneous variables to a
        function into a Drake cpst.
        
        The function is assumed to be called in the form `func(v1,v2,...,vn)`,
        unless pre_args is given, in which case it is assumed to be called
        `func(pre_args_1,...,pre_args_m,v1,v2,...,vn)`.

        This will handle all of the stacking and unstacking of the variables.
        """
        flat_func,flat_vars = self.wrap(func,vars,pre_args,post_args,kwargs)
        c = self.prog.AddCost(flat_func,flat_vars)
        if description:
            c.evaluator().set_description(description)

    def setup_program(self):
        prog = MathematicalProgram()
        self.prog = prog
        x = prog.NewContinuousVariables(rows=self.N, cols=3, name='x')
        q = prog.NewContinuousVariables(rows=self.N, cols=4, name='q')

        w = prog.NewContinuousVariables(rows=self.N, cols=3, name="w")
        v = prog.NewContinuousVariables(rows=self.N, cols=3, name='v')

        if self.param.velocity_complementarity:
            gamma_vars = {}
            dummy_vars = {}
            if self.param.use_alpha:
                alpha_vars = {}
            for i in range(self.N):
                k = len(self.iterate.states[i].index_set)
                gamma_vars[i]  = prog.NewContinuousVariables(k, name='gamma_'+str(i))
                dummy_vars[i] = prog.NewContinuousVariables(k, name='dummy_'+str(i))
                if self.param.use_alpha:
                    alpha_vars[i]  = prog.NewContinuousVariables(len(self.problem.manipulation_contact_points), name='alpha_'+str(i))
        
        force_vars = {}
        for i in range(self.N):
            k = len(self.iterate.states[i].index_set)
            force_vars['fenv_'+str(i)] = prog.NewContinuousVariables(rows=k,cols=self.friction_dim+1, name='fenv_'+str(i))

        for i in range(self.N):
            k = len(self.problem.manipulation_contact_points)
            force_vars['fmnp_'+str(i)] = prog.NewContinuousVariables(rows=k,cols=self.friction_dim+1, name='fmnp_'+str(i))

        d_vars = {}
        for i in range(self.N):
            k = len(self.iterate.states[i].index_set)
            d_vars[i] = prog.NewContinuousVariables(k, name='d_'+str(i))

        for i in range(self.N):
            prog.AddConstraint(lambda x: [x@x], [1], [1], q[i]).evaluator().set_description(f"q[{i}] unit quaternion constraint")
        for i in range(1, self.N):
            self.add_constraint(self.backward_euler_q,[0.0]*4,[0.0]*4,[q[i], q[i-1], w[i]],f"q[{i}] backward euler constraint",kwargs={'dt':self.problem.time_step})
            self.add_constraint(self.backward_euler_x,[0.0]*3,[0.0]*3,[x[i], x[i-1], v[i]],f"x[{i}] backward euler constraint",kwargs={'dt':self.problem.time_step})            
            #prog.AddConstraint(lambda q_qprev_v, dt=self.problem.time_step : self.backward_euler_q(*(np.split(q_qprev_v,[4,8]) + [dt])),lb=[0.0]*4, ub=[0.0]*4,vars=np.concatenate([q[i], q[i-1], w[i]])).evaluator().set_description(f"q[{i}] backward euler constraint")
            #prog.AddConstraint(lambda x_xprev_v, dt=self.problem.time_step : self.backward_euler_x(*(np.split(x_xprev_v,[3,6]) + [dt])),lb=[0.0]*3, ub=[0.0]*3,vars=np.concatenate([x[i], x[i-1], v[i]])).evaluator().set_description(f"x[{i}] backward euler constraint")
            
        for i in range(self.N):
            # Bounds on variables
            prog.AddBoundingBoxConstraint(self.problem.x_bound[0],self.problem.x_bound[1],x[i])
            prog.AddBoundingBoxConstraint(self.problem.v_bound[0],self.problem.v_bound[1],v[i]) 
            if np.isfinite(self.problem.w_max):
                prog.AddConstraint(lambda w: [np.dot(w,w)], lb=[0], ub=[self.problem.w_max**2], vars=w[i]).evaluator().set_description(f"Bound for w at {i}-th time step")
            #prog.AddBoundingBoxConstraint(-self.problem.w_max,self.problem.w_max,w_mag[i])

            # Constraints on environment contacts
            for j,point in enumerate(self.iterate.states[i].index_set):

                # Non-penetration constraint
                self.add_constraint(self.point_distance_constraint,pre_args=(point,),lb=[0.],ub=[0.],vars=[q[i],x[i],d_vars[i][j]],description=f"Non-penetration for {j}-th point at {i}-th time step")
                #prog.AddConstraint(lambda qxd:self.point_distance_constraint(point,*np.split(qxd, [4, 4+3])),lb=[0.],ub=[0.],vars=np.concatenate([q[i],x[i],[d_vars[i][j]]])).evaluator().set_description(f"Non-penetration for {j}-th point at {i}-th time step")
                prog.AddBoundingBoxConstraint(0,np.inf,d_vars[i][j]).evaluator().set_description(f"Dummy non-penetration for {j}-th point at {i}-th time step")
                
                # Bounds on environment contact forces
                lb = [0]*(self.friction_dim+1)
                ub = [self.problem.fenv_max]+[np.inf]*self.friction_dim
                prog.AddBoundingBoxConstraint(lb,ub,force_vars['fenv_'+str(i)][j]).evaluator().set_description(f"Bound for {j}-th point at {i}-th time step")

                # Complementarity constraint on environment contacts
                prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param.comp_threshold],vars=[force_vars['fenv_'+str(i)][j][0],d_vars[i][j]]).evaluator().set_description(f"CC for {j}-th point at {i}-th time step")
                
                # Friction cone constraint on envionment contacts
                prog.AddConstraint(partial(self.friction_cone_constraint,self.problem.mu_env),lb=[0],ub=[np.inf],vars=force_vars['fenv_'+str(i)][j]).evaluator().set_description(f"Friction cone constraint for {j}-th env point at {i}-th time step")
                
            # Complementarity on velocity
            if self.param.velocity_complementarity:
                for j,point in enumerate(self.iterate.states[i].index_set): 
                    #k complementarity constraints and one positivity constraint
                    lb = [-np.inf]*self.friction_dim+[0.]*self.friction_dim
                    ub = [self.param.comp_threshold]*self.friction_dim+[np.inf]*self.friction_dim
                    self.add_constraint(self.velocity_cc_constraint,lb=lb,ub=ub,pre_args=(point,),vars=[q[i],w[i],x[i],v[i],gamma_vars[i][j],force_vars['fenv_'+str(i)][j]],description=f"Velocity CC constraint for {j}-th point at time step {i}.")
                    #prog.AddConstraint(partial(self.velocity_cc_constraint,point,k),lb=[-np.inf,0.],ub=[self.param.comp_threshold,np.inf],vars=np.concatenate([q[i],w[i],x[i],v[i],[gamma_vars[i][j]],[force_vars['fenv_'+str(i)][j][k+1]]])).evaluator().set_description(f"Velocity CC constraint for {j}-th point dim-{k} at time step {i}.")
                    self.add_constraint(self.dummy_friction_residual_constraint,lb=[0.],ub=[0.],pre_args=(self.problem.mu_env,),vars=[force_vars['fenv_'+str(i)][j],dummy_vars[i][j]],description=f"Gamma constraint for {j}-th point at {i}-th time step")
                    #prog.AddConstraint(lambda fdummy:self.dummy_friction_residual_constraint(self.problem.mu_env,*np.split(fdummy,[self.friction_dim+1])),lb=[0.],ub=[0.],vars=np.concatenate([force_vars['fenv_'+str(i)][j],[dummy_vars[i][j]]]))
                    prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param.comp_threshold],vars=[dummy_vars[i][j],gamma_vars[i][j]])
                    prog.AddBoundingBoxConstraint(0,np.inf,gamma_vars[i][j]).evaluator().set_description(f"Gamma constraint for {j}-th point at {i}-th time step")

            # Constraints on manipulation contacts
            for j in range(len(self.problem.manipulation_contact_points)):
                # Bounds on manipulation contact forces
                lb = [0]*(self.friction_dim+1)
                ub = [self.problem.fmnp_max]+[np.inf]*self.friction_dim
                prog.AddBoundingBoxConstraint(lb,ub,force_vars['fmnp_'+str(i)][j]).evaluator().set_description(f"Bound for {j}-th mnp point at {i}-th time step")

                # Friction cone constraint on manipulation contacts
                self.add_constraint(self.friction_cone_constraint,pre_args=(self.problem.mu_mnp,),lb=[0],ub=[np.inf],vars=force_vars['fmnp_'+str(i)][j],description=f"Friction cone constraint for {j}-th mnp point at {i}-th time step")
                #prog.AddConstraint(partial(self.friction_cone_constraint,self.problem.mu_mnp),lb=[0],ub=[np.inf],vars=force_vars['fmnp_'+str(i)][j]).evaluator().set_description(f"Friction cone constraint for {j}-th mnp point at {i}-th time step")
                
                # Force on and off
                if self.param.force_on_off:
                    if self.param.use_alpha:
                        prog.AddConstraint(lambda qxfa:self.force_on_off(self.problem.manipulation_contact_points[j],*np.split(qxfa,[4, 4+3, 7+1])),lb=[4e-2,0],ub=[np.inf,0],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]],alpha_vars[i][j]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")    
                    else:
                        prog.AddConstraint(lambda qxf:self.force_on_off(self.problem.manipulation_contact_points[j],np.split(qxf,[4,4+3])),lb=[0.],ub=[self.param.comp_threshold],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")
                                
            # Balance constraints
            #fvecs = [force_vars['fenv_'+str(i)][j] for j in range(len(self.iterate.states[i].index_set))]+[force_vars['fmnp_'+str(i)][j] for j in range(len(self.problem.manipulation_contact_points))]
            self.add_constraint(self.force_balance_constraint,lb=[0.,0.,0.],ub=[0.,0.,0.],pre_args=(i,),vars=[q[i],x[i],v[i],force_vars['fenv_'+str(i)],force_vars['fmnp_'+str(i)]],description=f"Force balance at {i}-th time step")
            self.add_constraint(self.torque_balance_constraint,lb=[0.,0.,0.],ub=[0.,0.,0.],pre_args=(i,),vars=[q[i],x[i],w[i],force_vars['fenv_'+str(i)],force_vars['fmnp_'+str(i)]],description=f"Torque balance at {i}-th time step")
            #prog.AddConstraint(lambda qxvf,i=i:self.force_balance_constraint(i,*np.split(qxvf,[4,4+3,7+3,10+(self.friction_dim+1)*len(self.iterate.states[i].index_set)])),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],v[i]]+fvecs)).evaluator().set_description(f"Force balance at {i}-th time step")
            #prog.AddConstraint(lambda qxwf,i=i:self.torque_balance_constraint(i,*np.split(qxwf,[4,4+3,7+3,10+(self.friction_dim+1)*len(self.iterate.states[i].index_set)])),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],w[i]]+fvecs)).evaluator().set_description(f"Torque balance at {i}-th time step")
        
        # Boundary constraints
        initial_pose_relaxation = self.problem.initial_pose_relaxation
        goal_pose_relaxation = self.problem.goal_pose_relaxation
        q_init = so3.quaternion(self.problem.T_init[0])
        x_init = self.problem.T_init[1]
        q_goal = so3.quaternion(self.problem.T_goal[0])
        x_goal = self.problem.T_goal[1]
        for i in range(4):
            prog.AddBoundingBoxConstraint(q_init[i]-initial_pose_relaxation, q_init[i]+initial_pose_relaxation, q[0][i]).evaluator().set_description(f"Initial orientation constraint {i}")
            prog.AddBoundingBoxConstraint(q_goal[i]-goal_pose_relaxation, q_goal[i]+goal_pose_relaxation, q[-1][i]).evaluator().set_description(f"Goal orientation constraint {i}")
        for i in range(3):
            prog.AddBoundingBoxConstraint(x_init[i]-initial_pose_relaxation, x_init[i]+initial_pose_relaxation, x[0][i]).evaluator().set_description(f"Initial position constraint {i}")
            prog.AddBoundingBoxConstraint(x_goal[i]-goal_pose_relaxation, x_goal[i]+goal_pose_relaxation, x[-1][i]).evaluator().set_description(f"Goal position constraint {i}")

        # Objective function
        self.add_cost(self.cost_fun,vars=[q,x])
        
        # Set up initial guess for the optimization
        for i in range(self.N):
            prog.SetInitialGuess(x[i], self.iterate.states[i].x)
            prog.SetInitialGuess(q[i], self.iterate.states[i].q)
            prog.SetInitialGuess(w[i], self.iterate.states[i].w)
            prog.SetInitialGuess(v[i], self.iterate.states[i].v)

            for j in range(len(self.problem.manipulation_contact_points)):
                prog.SetInitialGuess(force_vars['fmnp_'+str(i)][j], self.iterate.states[i].fmnp[j])
                if self.param.use_alpha:
                    prog.SetInitialGuess(alpha_vars[i][j], [self.iterate.states[i].alpha[j]])

            for j in range(len(self.iterate.states[i].index_set)):
                prog.SetInitialGuess(d_vars[i][j], self.iterate.states[i].d[j])
                prog.SetInitialGuess(force_vars['fenv_'+str(i)][j], self.iterate.states[i].fenv[j])
                if self.param.velocity_complementarity:
                    prog.SetInitialGuess(gamma_vars[i][j], self.iterate.states[i].gamma[j])
                    prog.SetInitialGuess(dummy_vars[i][j], self.iterate.states[i].dummy[j])

        self.q = q
        self.x = x
        self.w = w
        self.v = v
        self.force_vars = force_vars
        self.d_vars = d_vars
        if self.param.velocity_complementarity:
            self.gamma_vars = gamma_vars
            self.dummy_vars = dummy_vars
        if self.param.use_alpha:
            self.alpha_vars = alpha_vars

    def solve(self) -> TrajectoryState:
        if self.prog is None:
            self.setup_program()
        prog = self.prog
        solver = SnoptSolver()
        snopt = solver.solver_id()
        prog.SetSolverOption(snopt, "Major Iterations Limit", self.param.max_mpcc_iter)
        prog.SetSolverOption(snopt, "Major Feasibility Tolerance", self.param.major_feasibility_tolerance)
        prog.SetSolverOption(snopt, "Major Optimality Tolerance", self.param.major_optimality_tolerance)

        filename = "tmp/debug.txt"
        prog.SetSolverOption(snopt, 'Print file', filename)
        result = solver.Solve(prog)

        #parse solution
        qs = result.GetSolution(self.q)
        qs = [q/np.linalg.norm(q) for q in qs]
        xs = result.GetSolution(self.x)
        ws = result.GetSolution(self.w)
        vs = result.GetSolution(self.v)
        new_states = []
        for i in range(self.N):
            new_states.append(replace(self.iterate.states[i],q=qs[i],x=xs[i],w=ws[i],v=vs[i]))
            new_states[-1].fenv = result.GetSolution(self.force_vars['fenv_'+str(i)])
            new_states[-1].fmnp = result.GetSolution(self.force_vars['fmnp_'+str(i)])
            new_states[-1].fenv = [f for f in new_states[-1].fenv]  #convert to list
            new_states[-1].fmnp = [f for f in new_states[-1].fmnp]  #convert to list
            new_states[-1].d = result.GetSolution(self.d_vars[i]).tolist()
            if self.param.velocity_complementarity:
                new_states[-1].gamma = result.GetSolution(self.gamma_vars[i]).tolist()
                new_states[-1].dummy = result.GetSolution(self.dummy_vars[i]).tolist()
            if self.param.use_alpha:
                new_states[-1].alpha = result.GetSolution(self.alpha_vars[i]).tolist()
        return TrajectoryState(new_states,self.iterate.times)


if __name__ == '__main__':
    # Take arguments from the command line
    import argparse
    import importlib
    import time
    import os

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
            from visualization import plot_problem_klampt
            plot_problem_klampt(problem)

    problem.manipuland_name = manipuland
    problem.environment_name = environment
    stocs_3d = STOCS(problem,params)

    res = stocs_3d.solve()
    print(f"Time used: {res.time}s.")
    print(f"Success: {res.is_success}")
    print(f"Outer iterations: {res.total_iter}")
    print(f"Average index point: {res.average_index_points()}")
    
    # Save the result as JSON
    import dataclasses
    import json
    folder = f'output/{environment}_{manipuland}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("Saving results to", folder)
    with open(f'{folder}/result.json', 'w') as f:
        json.dump(dataclasses.asdict(res),f)
    with open(f'{folder}/params.json', 'w') as f:
        json.dump(dataclasses.asdict(params),f)
    with open(f'{folder}/problem.json', 'w') as f:
        replaced_problem = replace(problem,manipuland=None,environments=[],environment_sdf_cache=None)
        json.dump(dataclasses.asdict(replaced_problem),f)