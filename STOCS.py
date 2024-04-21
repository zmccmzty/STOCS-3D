from __future__ import annotations
import numpy as np
import time
import copy
import math
from functools import partial
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import SE3Trajectory
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions, SnoptSolver, IpoptSolver, NloptSolver, CommonSolverOption
from pydrake.autodiffutils import AutoDiffXd
from semiinfinite.geometryopt import *
import threading
from utils import *
from typing import List,Tuple,Dict,Optional,Union
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
                vis.debug(environment_mesh,title="Environment SDF")


@dataclass
class SmoothingOracleParams:
    """Defines parameters for the spatio-temporal smoothing oracle."""
    add_threshold: float                    # Distance threshold to add active points
    remove_threshold: float                 # Distance threshold to remove active points.  Should be greater than add_threshold
    translation_disturbances: Optional[List[float]]=None   # Disturbances for spatial smoothing
    rotation_disturbances: Optional[List[float]]=None      # Disturbances for spatial smoothing
    time_step: int = 1                      # Time step for temporal smoothing. 1 gives reasonable results
    duplicate_detection_threshold: float = 1e-3    # Threshold for detecting duplicate points


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
        return (so3.from_quaternion(self.q),self.x.tolist())

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
        x = vectorops.interpolate(a.x,b.x,u)
        q = so3.quaternion(so3.interpolate(so3.from_quaternion(a.q),so3.from_quaternion(b.q),u))
        w = vectorops.interpolate(a.w,b.w,u)
        v = vectorops.interpolate(a.v,b.v,u)
        index_set = [vectorops.interpolate(a.index_set[i],b.index_set[i],u) for i in range(len(a.index_set))]
        fenv = [vectorops.interpolate(a.fenv[i],b.fenv[i],u) for i in range(len(a.fenv))]
        fmnp = [vectorops.interpolate(a.fmnp[i],b.fmnp[i],u) for i in range(len(a.fmnp))]
        d = vectorops.interpolate(a.d,b.d,u)
        gamma = vectorops.interpolate(a.gamma,b.gamma,u)
        dummy = vectorops.interpolate(a.dummy,b.dummy,u)
        alpha = vectorops.interpolate(a.alpha,b.alpha,u)
        return TimestepState(x=x,q=q,v=v,w=w,index_set=index_set,fenv=fenv,fmnp=fmnp,d=d,gamma=gamma,dummy=dummy,alpha=alpha)


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
        fenv = [np.stack(s.fenv) for s in self.states]
        fmnp = [np.stack(s.fmnp) for s in self.states]
        return {'q':np.array(q),'x':np.array(x),'w':np.array(w),'v':np.array(v),'fenv':np.stack(fenv),'fmnp':np.stack(fmnp)}

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
        time_smoothing_step = self.optimization_params.oracle_params.time_step

        closest_points = [[] for _ in range(self.N)]
        for i,state in enumerate(iterate.states):           
            closest_points_ti = closest_points[ti]

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
                for point2 in iterate.states[ti].index_set:
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
                mindist = min(environment.distance(point)[0] for environment in self.problem.environments)
                if mindist > self.optimization_params.oracle_params.remove_threshold:
                    removed_points[ti].append(j)
            
        return new_points,removed_points

    def set_initialization(self, initial : Optional[Union[TrajectoryState,SE3Trajectory]] = None):
        """Initializes the optimizer according to the initialization parameter
        or with a given initial trajectory.
        """
        if initial is None:
            times = np.arange(0,self.problem.N*self.problem.time_step,self.problem.time_step)
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
                
                initial_states.append(TimestepState(x=T[1],q=so3.quaternion(T[0]),v=v,w=w,index_set=[],fenv=[],fmnp=[],d=[],gamma=[],dummy=[],alpha=[]))
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
            self.set_initialization()

        stocs_res = Result(is_success=False,total_iter=0,iterates=[],final=replace(self.current_iterate))
        
        while iter < self.optimization_params.stocs_max_iter:
            #call oracle
            new_index_set, removed_index_set = self.oracle(self.current_iterate)

            #update / initialize variables
            for i in range(self.N):
                statei = self.current_iterate.states[i]
                #remove index points and variables
                for j in removed_index_set[i]:
                    statei.index_set.pop(j)
                    statei.fenv.pop(j)
                    statei.fmnp.pop(j)
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
                    statei.fenv.append([1e-3] + [1e-3]*self.optimization_params.friction_dim)
                
                statei.d += [0.0]*len(new_index_set[i])
                
                if self.optimization_params.use_alpha:
                    statei.alpha += [0.0]*len(new_index_set[i])

                #add new index points
                statei.index_set += new_index_set[i]

            #update trace
            stocs_res.iterates.append(copy.deepcopy(self.current_iterate))
            stocs_res.final = stocs_res.iterates[-1]
            stocs_res.total_iter = iter
            print(f"Index points for each time step along the trajectory: {[len(sublist) for sublist in self.current_iterate.states]}")

            #TODO: THIS IS A HACK! need to make the iteration growth rate a proper parameter
            self.optimization_params.max_mpcc_iter = min(5+5*iter,100)
            mpcc = MPCC(self.problem,self.optimization_params,self.current_iterate)
            print(f"Start the solving of iteration {iter}")

            def print_dots(stop_event):
                while not stop_event.is_set():
                    print(".", end="", flush=True)
                    time.sleep(1)  

            stop_event = threading.Event()

            dot_thread = threading.Thread(target=print_dots, args=(stop_event,))
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
            diffs = [prev_flat[s_var] - new_flat[s_var] for s_var in ['q','x','w','v']]
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
        score = mpcc.cost_fun(res) + self._constraint_violation(mpcc,res)
        return score

    def _line_search(self, mpcc : MPCC,res_target : TrajectoryState, res_current : TrajectoryState):
        current_score = self._score_function(mpcc,res_current)
        target_score = self._score_function(mpcc,res_target)

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
            Tprev = mpcc.backward_euler(res.states[i],self.problem.time_step)
            violation += np.sum(se3.error(Tprev,res.states[i-1].pose()))

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
            for j in range(len(res.states[i].index_set)):
                # Position Comp
                residual += abs(min(self.optimization_params.comp_threshold-res.states[i].fenv[j][0]*res.states[i].d[j],0))
                
                # Velocity Comp
                if self.optimization_params.velocity_complementarity:
                    for k in range(self.optimization_params.friction_dim):
                        x_ = np.concatenate([res.states[i].q,res.states[i].w,res.states[i].x,res.states[i].v,res.states[i].gamma[j],res.states[i].fenv[j][k+1]])
                        constraint_value = partial(mpcc.velocity_cc_constraint,res.states[i].index_set,k)(x_)
                        residual += abs(min(self.optimization_params.comp_threshold - constraint_value[0],0))                   
                    residual += abs(min(self.optimization_params.comp_threshold - res.states[i].dummy[j]*res.states[i].gamma[j],0))

        return residual

    def _balance_residual(self,mpcc : MPCC, res : TrajectoryState):
        residual = 0
        for i in range(self.N):
            s = res.states[i]
            residual += np.sum(np.abs(mpcc.force_balance_constraint(i,s.q,s.x,s.v,np.concat(s.fmnp+s.fenv))))
            residual += np.sum(np.abs(mpcc.torque_balance_constraint(i,s.q,s.x,s.w,np.concat(s.fmnp+s.fenv)))) 

        return residual


class MPCC(object):
    def __init__(self,problem: Problem, param : OptimizerParams, iterate: TrajectoryState):

        self.problem = problem
        self.param = param
        self.iterate = iterate

        # Optimization parameters
        self.N = self.problem.N 
        self.initial_pose_relaxation = self.problem.initial_pose_relaxation
        self.goal_pose_relaxation = self.problem.goal_pose_relaxation
        self.comp_threshold = self.param.comp_threshold
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

        res = dist - d[0]

        return np.array([res])
        
    def force_balance_constraint(self, t : int, q, x, v, f):

        env_force_dim = (self.friction_dim+1)*len(self.iterate.states[t].index_set)
        f_env, f_mnp = np.split(f, [env_force_dim])

        force = 0
        # Environment contact 
        for i,point in enumerate(self.iterate.states[t].index_set):
            
            f_env_i = f_env[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            
            point_world = se3.apply((so3.from_quaternion(q),x),point)
            dir_n = np.asarray(self.environment_distance_gradient(point_world))
            N_normal_ = dir_n / np.linalg.norm(dir_n)

            assert np.linalg.norm(N_normal_) != 0, "Normal is 0!!!"
            N_normal = N_normal_ / np.linalg.norm(N_normal_)

            n1 = so3.canonical(N_normal)[3:6]
            n2 = so3.canonical(N_normal)[6:9]

            N_friction = []
            for j in range(self.friction_dim):
                n_tmp_ = math.cos((math.pi/int(self.friction_dim/2))*j)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*j)*np.array(n2)
                n_tmp = n_tmp_ / np.linalg.norm(n_tmp_)
                N_friction.append(n_tmp)

            force += f_env_i[0]*np.array(N_normal)

            for j in range(len(N_friction)):
                force += f_env_i[j+1]*np.array(N_friction[j])
        
        # Gravity
        force += np.array([0,0,-self.problem.gravity*self.problem.manipuland_mass])

        # Manipulator contact force
        for i,manipulation_Normal_normal in enumerate(self.problem.manipulation_contact_normals):
            
            f_mnp_i = f_mnp[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            manipulation_N_normal = so3.apply(so3.from_quaternion(q),manipulation_Normal_normal)
            manipulation_N_normal = list(np.array(manipulation_N_normal) / np.linalg.norm(manipulation_N_normal))
            
            force += f_mnp_i[0] * np.array(manipulation_N_normal)
            n1 = so3.canonical(manipulation_N_normal)[3:6]
            n2 = so3.canonical(manipulation_N_normal)[6:9]
            
            manipulation_N_friction = []
            for j in range(self.friction_dim):
                n_tmp = (math.cos((math.pi/int(self.friction_dim/2))*j)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*j)*np.array(n2)).tolist()   
                manipulation_N_friction.append(list(np.array(n_tmp) / np.linalg.norm(n_tmp)))
            for j,normal in enumerate(manipulation_N_friction):
                force += f_mnp_i[j+1]*np.array(normal)

        if self.param.assumption == 'quasi_static':
            return np.array([force[0],force[1],force[2]])
        elif self.param.assumption == 'quasi_dynamic':
            if self.dt[t] != 0:
                return force-self.problem.manipuland_mass*v/self.dt[t]
            else:
                return np.array([force[0],force[1],force[2]])

    def torque_balance_constraint(self,t : int, q, x, w, f):
        env_force_dim = (self.friction_dim+1)*len(self.iterate.states[t].index_set)
        f_env, f_mnp = np.split(f, [env_force_dim])

        com_world = se3.apply((so3.from_quaternion(q),x),self.problem.manipuland_com)
        torque = 0

        # Contact with the environment
        for i,point in enumerate(self.iterate.states[t].index_set):
            
            f_env_i = f_env[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            point_world = se3.apply((so3.from_quaternion(q),x),point)

            dir_n_ = np.asarray(self.environment_distance_gradient(point_world))
            N_normal = dir_n_ / np.linalg.norm(dir_n_)

            torque += np.array(vectorops.cross(vectorops.sub(point_world,com_world),vectorops.mul(N_normal,f_env_i[0])))

            n1 = so3.canonical(N_normal)[3:6]
            n2 = so3.canonical(N_normal)[6:9]
            N_friction = []
            for j in range(self.friction_dim):
                n_tmp = math.cos((math.pi/int(self.friction_dim/2))*j)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*j)*np.array(n2)
                N_friction.append(list(np.array(n_tmp) / np.linalg.norm(n_tmp)))
            for j in range(len(N_friction)):
                torque += np.array(vectorops.cross(vectorops.sub(point_world,com_world),vectorops.mul(N_friction[j],f_env_i[j+1])))
        
        # Contact with the manipulator
        for i,(manipulation_contact,manipulation_Normal_normal) in enumerate(zip(self.problem.manipulation_contact_points,self.problem.manipulation_contact_normals)):
            f_mnp_i = f_mnp[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            mnp_contact_world = se3.apply((so3.from_quaternion(q),x),manipulation_contact)
            manipulation_N_normal = so3.apply(so3.from_quaternion(q),manipulation_Normal_normal)
            manipulation_N_normal = list(np.array(manipulation_N_normal) / np.linalg.norm(manipulation_N_normal))

            torque += np.array(vectorops.cross(vectorops.sub(mnp_contact_world,com_world),vectorops.mul(so3.apply(so3.from_quaternion(q),manipulation_N_normal),f_mnp_i[0])))

            n1 = so3.canonical(manipulation_N_normal)[3:6]
            n2 = so3.canonical(manipulation_N_normal)[6:9]
    
            manipulation_N_friction = []
            for j in range(self.friction_dim):
                n_tmp = (math.cos((math.pi/int(self.friction_dim/2))*j)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*j)*np.array(n2)).tolist()
                manipulation_N_friction.append(list(np.array(n_tmp) / np.linalg.norm(n_tmp)))
            for j in range(len(manipulation_N_friction)):
                torque += np.array(vectorops.cross(vectorops.sub(mnp_contact_world,com_world),vectorops.mul(so3.apply(so3.from_quaternion(q),manipulation_N_friction[j]),f_mnp_i[j+1])))

        if self.param.assumption == 'quasi_static':
            return np.array([torque[0],torque[1],torque[2]])
        elif self.param.assumption == 'quasi_dynamic':
            I_body = self.problem.manipuland_inertia
            R = np.array(so3.matrix(so3.from_quaternion(q)))
            I_world = R@I_body@R.T
            if self.dt[t] != 0:
                return torque - I_world@w/self.dt[t]
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

    def force_on_off(self,point,q, x, f, alpha=None):
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

    def velocity_cc_constraint(self,point,i,x_):
        
        q,w_axis,w_mag,x,v,gamma,f = np.split(x_, [4, 4+3, 7+1, 8+3, 11+3, 14+1])
        v_relative = vectorops.cross(list(np.array(w_axis)*w_mag),so3.apply(so3.from_quaternion(q),point))
        v_real = vectorops.add(v_relative,v)

        point_world = se3.apply((so3.from_quaternion(q),x),point)
        dir_n_ = np.asarray(self.environment_distance_gradient(point_world))
        N_normal = dir_n_ / np.linalg.norm(dir_n_)
        
        v_tangential = vectorops.sub(v_real,vectorops.mul(N_normal,vectorops.dot(v_real,N_normal)))

        n1 = so3.canonical(N_normal)[3:6]
        n2 = so3.canonical(N_normal)[6:9]

        n_tmp = (math.cos((math.pi/int(self.friction_dim/2))*i)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*i)*np.array(n2)).tolist()
        N_friction = list(np.array(n_tmp) / np.linalg.norm(n_tmp)) 
        res1 = (gamma + vectorops.dot(v_tangential,N_friction))*f
        res2 = gamma + vectorops.dot(v_tangential,N_friction)
        return np.array([res1[0],res2[0]])

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
        res -= residual[0]
        return np.array([res])    

    def cost_fun(self,x_):
        res = 0*x_[0]
        return res

    def setup_program(self):
        prog = MathematicalProgram()
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
            prog.AddConstraint(lambda q_qprev_v, dt=self.problem.time_step : self.backward_euler_q(*(np.split(q_qprev_v,[4,4]) + [dt])),lb=[0.0]*4, ub=[0.0]*4,vars=np.concat([q[i], q[i-1], w[i]])).evaluator().set_description(f"q[{i}] backward euler constraint")
            prog.AddConstraint(lambda x_xprev_v, dt=self.problem.time_step : self.backward_euler_x(*(np.split(x_xprev_v,[3,6]) + [dt])),lb=[0.0]*3, ub=[0.0]*3,vars=np.concat([x[i], x[i-1], v[i]])).evaluator().set_description(f"x[{i}] backward euler constraint")

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
                prog.AddConstraint(lambda qxd:self.point_distance_constraint(point,*np.split(qxd, [4, 4+3])),lb=[0.],ub=[0.],vars=np.concat([q[i],x[i],d_vars[i][j]])).evaluator().set_description(f"Non-penetration for {j}-th point at {i}-th time step")
                prog.AddBoundingBoxConstraint(0,np.inf,d_vars[i][j]).evaluator().set_description(f"Dummy non-penetration for {j}-th point at {i}-th time step")
                
                # Bounds on environment contact forces
                prog.AddBoundingBoxConstraint(0,self.problem.fenv_max,force_vars['fenv_'+str(i)][j][0])
                for k in range(1,self.friction_dim+1):
                    prog.AddBoundingBoxConstraint(0,np.inf,force_vars['fmnp_'+str(i)][j][k]).evaluator().set_description(f"Bound for {j}-th mnp point at {i}-th time step")

                # Complementarity constraint on environment contacts
                prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param.comp_threshold],vars=[force_vars['fenv_'+str(i)][j][0],d_vars[i][j]]).evaluator().set_description(f"CC for {j}-th point at {i}-th time step")

                # Friction cone constraint on envionment contacts
                prog.AddConstraint(partial(self.friction_cone_constraint,self.problem.mu_env),lb=[0],ub=[np.inf],vars=force_vars['fenv_'+str(i)][j]).evaluator().set_description(f"Friction cone constraint for {j}-th env point at {i}-th time step")

            # Complementarity on velocity
            if self.param.velocity_complementarity:
                for j,point in enumerate(self.iterate.states[i].index_set): 
                    for k in range(self.friction_dim):
                        prog.AddConstraint(partial(self.velocity_cc_constraint,point,k),lb=[-np.inf,0.],ub=[self.param.comp_threshold,np.inf],vars=np.concatenate([q[i],w[i],x[i],v[i],gamma_vars[i][j],[force_vars['fenv_'+str(i)][j][k+1]]])).evaluator().set_description(f"Velocity CC constraint for {j}-th point dim-{k} at time step {i}.")
                    prog.AddConstraint(lambda fdummy:self.dummy_friction_residual_constraint(self.problem.mu_env,np.split(fdummy,[self.friction_dim+1])),lb=[0.],ub=[0.],vars=np.concat([force_vars['fenv_'+str(i)][j],dummy_vars[i][j]]))
                    prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param.comp_threshold],vars=[dummy_vars[i][j],gamma_vars[i][j]])
                    prog.AddBoundingBoxConstraint(0,np.inf,gamma_vars[i][j]).evaluator().set_description(f"Gamma constraint for {j}-th point at {i}-th time step")

            # Constraints on manipulation contacts
            for j in range(len(self.problem.manipulation_contact_points)):
                # Bounds on manipulation contact forces
                prog.AddBoundingBoxConstraint(0,self.problem.fmnp_max,force_vars['fmnp_'+str(i)][j][0])
                for k in range(1,self.friction_dim+1):
                    prog.AddBoundingBoxConstraint(0,np.inf,force_vars['fmnp_'+str(i)][j][k]).evaluator().set_description(f"Bound for {j}-th mnp point at {i}-th time step")

                # Friction cone constraint on manipulation contacts
                prog.AddConstraint(partial(self.friction_cone_constraint,self.problem.mu_mnp),lb=[0],ub=[np.inf],vars=force_vars['fmnp_'+str(i)][j]).evaluator().set_description(f"Friction cone constraint for {j}-th mnp point at {i}-th time step")

                # Forece on and off
                if self.param.force_on_off:
                    if self.param.use_alpha:
                        prog.AddConstraint(lambda qxfa:self.force_on_off(self.problem.manipulation_contact_points[j],*np.split(qxfa,[4, 4+3, 7+1])),lb=[4e-2,0],ub=[np.inf,0],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]],alpha_vars[i][j]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")    
                    else:
                        prog.AddConstraint(lambda qxf:self.force_on_off(self.problem.manipulation_contact_points[j],np.split(qxf,[4,4+3])),lb=[0.],ub=[self.param.comp_threshold],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)][j][0]]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")
                                
            # Balance constraints
            prog.AddConstraint(lambda qxvf:self.force_balance_constraint(i,np.split(qxvf,[4,3,3])),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],v[i]]+[force_vars['fenv_'+str(i)][j] for j in range(len(self.iterate.states[i].index_set))]+[force_vars['fmnp_'+str(i)][j] for j in range(len(self.problem.manipulation_contact_points))])).evaluator().set_description(f"Force balance at {i}-th time step")
            prog.AddConstraint(lambda qxwf:self.torque_balance_constraint(i,np.split(qxwf,[4,3,3])),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],w[i]]+[force_vars['fenv_'+str(i)][j] for j in range(len(self.iterate.states[i].index_set))]+[force_vars['fmnp_'+str(i)][j] for j in range(len(self.problem.manipulation_contact_points))])).evaluator().set_description(f"Torque balance at {i}-th time step")

        # Boundary constraints
        initial_pose_relaxation = self.param.initial_pose_relaxation
        goal_pose_relaxation = self.param.goal_pose_relaxation
        for i in range(4):
            prog.AddBoundingBoxConstraint(self.q_init[i]-initial_pose_relaxation, self.q_init[i]+initial_pose_relaxation, q[0][i]).evaluator().set_description(f"Initial orientation constraint {i}")
            prog.AddBoundingBoxConstraint(self.q_goal[i]-goal_pose_relaxation, self.q_goal[i]+goal_pose_relaxation, q[-1][i]).evaluator().set_description(f"Goal orientation constraint {i}")
        for i in range(3):
            prog.AddBoundingBoxConstraint(self.x_init[i]-initial_pose_relaxation, self.x_init[i]+initial_pose_relaxation, x[0][i]).evaluator().set_description(f"Initial position constraint {i}")
            prog.AddBoundingBoxConstraint(self.x_goal[i]-goal_pose_relaxation, self.x_goal[i]+goal_pose_relaxation, x[-1][i]).evaluator().set_description(f"Goal position constraint {i}")

        # Objective function
        prog.AddCost(self.cost_fun,vars=np.array([q[i] for i in range(self.N)]+[x[i] for i in range(self.N)]))
        
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

        self.prog = prog
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
        xs = result.GetSolution(self.x)
        ws = result.GetSolution(self.w)
        vs = result.GetSolution(self.v)
        new_states = []
        for i in range(self.N):
            new_states.append(replace(self.iterate.states[i],q=qs[i],x=xs[i],w=ws[i],v=vs[i]))
            new_states[-1].fenv = result.GetSolution(self.force_vars['fenv_'+str(i)])
            new_states[-1].fmnp = result.GetSolution(self.force_vars['fmnp_'+str(i)])
            new_states[-1].d = result.GetSolution(self.d_vars[i])
            if self.param.velocity_complementarity:
                new_states[-1].gamma = result.GetSolution(self.gamma_vars[i])
                new_states[-1].dummy = result.GetSolution(self.dummy_vars[i])
            if self.param.use_alpha:
                new_states[-1].alpha = result.GetSolution(self.alpha_vars[i])
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
    importlib.reload(module)
    # Access the 'problem' and 'params' attribute from the imported module
    problem = getattr(module, 'problem', None)
    params = getattr(module, 'params', None)

    if problem is None or not isinstance(problem, Problem):
        print(f"The 'problem' attribute was not found in the module {module_name} or is not a Problem instance")
    elif params is None or not isinstance(params, OptimizerParams):
        print(f"The 'params' attribute was not found in the module {module_name} or is not a OptimizerParams instance")
    else:
        print(f"Successfully imported 'params' from {module_name}")
        print(f"Environment: {environment}")
        print(f"Manipuland: {manipuland}")
        print(f"Number of time steps: {problem.N}")
        print(f"Time step: {problem.time_step}")
        if parser.debug:
            print(f"Debug mode is on")
            from visualization import plot_problem_klampt
            plot_problem_klampt(problem)

    stocs_3d = STOCS(params)

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