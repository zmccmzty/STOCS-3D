from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory
from klampt import vis

import numpy as np
import sys
import time
import copy
from semiinfinite.geometryopt import *
import math
from functools import partial
import argparse
import importlib
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions, SnoptSolver, IpoptSolver, NloptSolver, CommonSolverOption
from pydrake.autodiffutils import AutoDiffXd
from utils import *

class MPCC(object):
    def __init__(self,index_set,param,prev_sol=None):

        self.index_set = index_set
        self.param = param
        self.prev_sol = prev_sol

        # Optimization parameters
        self.N = self.param['optimization_params']['N'] 
        self.initial_pose_relaxation = self.param['optimization_params']['initial_pose_relaxation']
        self.goal_pose_relaxation = self.param['optimization_params']['goal_pose_relaxation']
        self.comp_threshold = self.param['optimization_params']['comp_threshold']
        self.friction_dim = self.param['optimization_params']['friction_dim']
        self.mu_env = self.param['optimization_params']['mu_env']
        self.mu_mnp = self.param['optimization_params']['mu_mnp']

        # Task parameters
        self.q_init = self.param['task_params']['q_init']
        self.x_init = self.param['task_params']['x_init']
        self.q_goal = self.param['task_params']['q_goal']
        self.x_goal = self.param['task_params']['x_goal']

        self.environment_sdf_cache = self.param['environment_params']['environment_sdf_cache']  # type: SDFCache

    
    def point_distance_constraint(self,point_,x_):

        q, x, d = np.split(x_, [4, 4+3])
        point, env_idx = point_[:3], point_[3]

        point_world = se3.apply((so3.from_quaternion(q),x),point)     
        dist = self.environment_sdf_cache.distance(point_world)

        res = dist - d[0]

        return np.array([res])
        
    def force_balance_constraint(self,t,x_):

        env_force_dim = (self.friction_dim+1)*len(self.index_set[t])
        q, x, v, f_env, f_mnp = np.split(x_, [4, 4+3, 7+3, 10+env_force_dim])

        force = 0
        # Environment contact 
        for i,point in enumerate(self.index_set[t]):
            
            f_env_i = f_env[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            
            point_world = se3.apply((so3.from_quaternion(q),x),point)
            dir_n_ = np.array(self.environment_sdf_cache.gradient(point_world))
            N_normal = dir_n_ / np.linalg.norm(dir_n_)
            assert np.linalg.norm(N_normal) != 0, "Normal is 0!!!"

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
        force += np.array([0,0,-9.8*self.param['manipuland_params']['m']])

        # Manipulator contact force
        for i,manipulation_Normal_normal in enumerate(self.param['task_params']['manipulation_contact_normals']):
            
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

        if self.param['optimization_params']['assumption'] == 'quasi_static':
            return np.array([force[0],force[1],force[2]])
        elif self.param['optimization_params']['assumption'] == 'quasi_dynamic':
            if self.dt[t] != 0:
                return force-self.param['manipuland_params']['m']*v/self.dt[t]
            else:
                return np.array([force[0],force[1],force[2]])

    def torque_balance_constraint(self,t,x_):
        
        env_force_dim = (self.friction_dim+1)*len(self.index_set[t])
        q, x, w_axis, w_mag, f_env, f_mnp = np.split(x_, [4, 4+3, 7+3, 10+1, 11+env_force_dim])

        com_world = se3.apply((so3.from_quaternion(q),x),self.param['manipuland_params']['com'])
        torque = 0

        # Contact with the environment
        for i,point in enumerate(self.index_set[t]):
            
            f_env_i = f_env[(self.friction_dim+1)*i:(self.friction_dim+1)*(i+1)]
            point_world = se3.apply((so3.from_quaternion(q),x),point)

            dir_n_ = np.array(self.environment_sdf_cache.gradient(point_world))
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
        for i,(manipulation_contact,manipulation_Normal_normal) in enumerate(zip(self.param['task_params']['manipulation_contact_points'],self.param['task_params']['manipulation_contact_normals'])):
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

        if self.param['optimization_params']['assumption'] == 'quasi_static':
            return np.array([torque[0],torque[1],torque[2]])
        elif self.param['optimization_params']['assumption'] == 'quasi_dynamic':
            I_body = self.param['manipuland_params']['I']
            R = np.array(so3.matrix(so3.from_quaternion(q)))
            w = w_mag*w_axis
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

    def apply_angular_velocity_to_quaternion(self, q, w_axis, w_mag, t):
        delta_q = np.hstack([np.cos(w_mag* t/2.0), w_axis*np.sin(w_mag* t/2.0)])
        return  self.quat_multiply(q, delta_q)

    def backward_euler_q(self, q_qprev_v, dt):
        q, qprev, w_axis, w_mag = np.split(q_qprev_v, [4, 4+4, 8+3])
        return q - self.apply_angular_velocity_to_quaternion(qprev, w_axis, w_mag, dt)

    def backward_euler_x(self, x_xprev_v, dt):
        x, xprev, v = np.split(x_xprev_v, [3, 3+3])
        return x - (xprev + v*dt)

    def force_on_off(self,point,x_):
        if self.param['optimization_params']['use_alpha']:
            q, x, f, alpha = np.split(x_, [4, 4+3, 7+1])
            point_world = se3.apply((so3.from_quaternion(q),x),point)
            query_object = self.query_object_ad if isinstance(x_[0], AutoDiffXd) else self.query_object_f
            dist = query_object.ComputeSignedDistanceToPoint(p_WQ=point_world)[0].distance

            return np.array([dist+alpha,alpha*f])
        else:
            q, x, f = np.split(x_, [4, 4+3])
            point_world = se3.apply((so3.from_quaternion(q),x),point)   
            query_object = self.query_object_ad if isinstance(x_[0], AutoDiffXd) else self.query_object_f   
            dist = query_object.ComputeSignedDistanceToPoint(p_WQ=point_world)[0].distance
        
            return np.array([np.log(1+np.exp(-1000*dist))*f])

    def velocity_cc_constraint(self,point,i,x_):
        
        q,w_axis,w_mag,x,v,gamma,f = np.split(x_, [4, 4+3, 7+1, 8+3, 11+3, 14+1])
        v_relative = vectorops.cross(list(np.array(w_axis)*w_mag),so3.apply(so3.from_quaternion(q),point))
        v_real = vectorops.add(v_relative,v)

        point_world = se3.apply((so3.from_quaternion(q),x),point)
        dir_n_ = np.array(self.environment_sdf_cache.gradient(point_world))
        N_normal = dir_n_ / np.linalg.norm(dir_n_)
        
        v_tangential = vectorops.sub(v_real,vectorops.mul(N_normal,vectorops.dot(v_real,N_normal)))

        n1 = so3.canonical(N_normal)[3:6]
        n2 = so3.canonical(N_normal)[6:9]

        n_tmp = (math.cos((math.pi/int(self.friction_dim/2))*i)*np.array(n1) + math.sin((math.pi/int(self.friction_dim/2))*i)*np.array(n2)).tolist()
        N_friction = list(np.array(n_tmp) / np.linalg.norm(n_tmp)) 
        res1 = (gamma + vectorops.dot(v_tangential,N_friction))*f
        res2 = gamma + vectorops.dot(v_tangential,N_friction)
        return np.array([res1[0],res2[0]])

    def friction_cone_constraint(self,mu,x_):
        fn,ff = np.split(x_, [1])
        res = mu*fn 
        for ffi in ff:
            res -= ffi
        return np.array([res])

    def dummy_friction_residual_constraint(self,mu,x_):
        fn,ff,residual = np.split(x_, [1,self.friction_dim+1])
        res = mu*fn
        for ffi in ff:
            res -= ffi
        res -= residual
        return np.array([res])    

    def cost_fun(self,x_):

        res = 0*x_[0]
        return res

    def solve(self):
        prog = MathematicalProgram()

        x = prog.NewContinuousVariables(rows=self.N, cols=3, name='x')
        q = prog.NewContinuousVariables(rows=self.N, cols=4, name='q')

        w_axis = prog.NewContinuousVariables(rows=self.N, cols=3, name="w_axis")
        w_mag = prog.NewContinuousVariables(rows=self.N, cols=1, name="w_mag")
        v = prog.NewContinuousVariables(rows=self.N, cols=3, name='v')

        if self.param['optimization_params']['velocity_complementarity']:
            gamma_vars = {}
            dummy_vars = {}
            if self.param['optimization_params']['use_alpha']:
                alpha_vars = {}
            for i in range(self.N):
                for j in range(len(self.index_set[i])):
                    gamma_vars['gamma_'+str(i)+'_'+str(j)]  = prog.NewContinuousVariables(1, name='gamma_'+str(i)+'_'+str(j))
                    dummy_vars['dummy_'+str(i)+'_'+str(j)]  = prog.NewContinuousVariables(1, name='dummy_'+str(i)+'_'+str(j))
                if self.param['optimization_params']['use_alpha']:
                    for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                        alpha_vars['alpha_'+str(i)+'_'+str(j)]  = prog.NewContinuousVariables(1, name='alpha_'+str(i)+'_'+str(j))
        
        self.dt = [0.0]+[self.param['optimization_params']['time_step']]*(self.N-1) 

        force_vars = {}
        for i in range(self.N):
            for j in range(len(self.index_set[i])):
                force_vars['fenv_'+str(i)+'_'+str(j)] = prog.NewContinuousVariables(self.friction_dim+1, name='fenv_'+str(i)+'_'+str(j))

        for i in range(self.N):
            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                force_vars['fmnp_'+str(i)+'_'+str(j)] = prog.NewContinuousVariables(self.friction_dim+1, name='fmnp_'+str(i)+'_'+str(j))

        d_vars = {}
        for i in range(self.N):
            for j in range(len(self.index_set[i])):
                d_vars['d_'+str(i)+'_'+str(j)] = prog.NewContinuousVariables(1, name='d_'+str(i)+'_'+str(j))

        for i in range(self.N):
            prog.AddConstraint(lambda x: [x@x], [1], [1], q[i]).evaluator().set_description(f"q[{i}] unit quaternion constraint")
            (prog.AddConstraint(lambda x: [x@x], [1], [1], w_axis[i]).evaluator().set_description(f"w_axis[{i}] unit axis constraint"))
        for i in range(1, self.N):
            (prog.AddConstraint(lambda q_qprev_v, dt=self.dt[i] : self.backward_euler_q(q_qprev_v, dt),lb=[0.0]*4, ub=[0.0]*4,vars=np.concatenate([q[i], q[i-1], w_axis[i], w_mag[i]])).evaluator().set_description(f"q[{i}] backward euler constraint"))
            (prog.AddConstraint(lambda x_xprev_v, dt=self.dt[i] : self.backward_euler_x(x_xprev_v, dt),lb=[0.0]*3, ub=[0.0]*3,vars=np.concatenate([x[i], x[i-1], v[i]])).evaluator().set_description(f"x[{i}] backward euler constraint"))

        for i in range(self.N):

            # Bounds on variables
            prog.AddBoundingBoxConstraint(self.param['task_params']['x_bound'][0][0],self.param['task_params']['x_bound'][1][0],x[i][0])
            prog.AddBoundingBoxConstraint(self.param['task_params']['x_bound'][0][1],self.param['task_params']['x_bound'][1][1],x[i][1])
            prog.AddBoundingBoxConstraint(self.param['task_params']['x_bound'][0][2],self.param['task_params']['x_bound'][1][2],x[i][2])
            prog.AddBoundingBoxConstraint(self.param['task_params']['v_bound'][0][0],self.param['task_params']['v_bound'][1][0],v[i][0])  
            prog.AddBoundingBoxConstraint(self.param['task_params']['v_bound'][0][1],self.param['task_params']['v_bound'][1][1],v[i][1])
            prog.AddBoundingBoxConstraint(self.param['task_params']['v_bound'][0][2],self.param['task_params']['v_bound'][1][2],v[i][2])
            prog.AddBoundingBoxConstraint(-np.pi,np.pi,w_mag[i])

            # Constraints on environment contacts
            for j,point in enumerate(self.index_set[i]):

                # Non-penetration constraint
                prog.AddConstraint(partial(self.point_distance_constraint,point),lb=[0.],ub=[0.],vars=np.concatenate([q[i],x[i],d_vars['d_'+str(i)+'_'+str(j)]])).evaluator().set_description(f"Non-penetration for {j}-th point at {i}-th time step")
                prog.AddConstraint(lambda z: [z[0]],lb=[0.],ub=[np.inf],vars=[d_vars['d_'+str(i)+'_'+str(j)]]).evaluator().set_description(f"Dummy non-penetration for {j}-th point at {i}-th time step")
                
                # Bounds on environment contact forces
                prog.AddBoundingBoxConstraint(0,10,force_vars['fenv_'+str(i)+'_'+str(j)][0])
                for k in range(1,self.friction_dim+1):
                    prog.AddBoundingBoxConstraint(0,10,force_vars['fenv_'+str(i)+'_'+str(j)][k])
                
                # Complementarity constraint on environment contacts
                prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param['optimization_params']['comp_threshold']],vars=[force_vars['fenv_'+str(i)+'_'+str(j)][0],d_vars['d_'+str(i)+'_'+str(j)][0]]).evaluator().set_description(f"CC for {j}-th point at {i}-th time step")

                # Friction cone constraint on envionment contacts
                prog.AddConstraint(partial(self.friction_cone_constraint,self.param['optimization_params']['mu_env']),lb=[0],ub=[np.inf],vars=force_vars['fenv_'+str(i)+'_'+str(j)]).evaluator().set_description(f"Friction cone constraint for {j}-th env point at {i}-th time step")

            # Complementarity on velocity
            if self.param['optimization_params']['velocity_complementarity']:
                for j,point in enumerate(self.index_set[i]): 
                    for k in range(self.friction_dim):
                        prog.AddConstraint(partial(self.velocity_cc_constraint,point,k),lb=[-np.inf,0.],ub=[self.param['optimization_params']['comp_threshold'],np.inf],vars=np.concatenate([q[i],w_axis[i],w_mag[i],x[i],v[i],gamma_vars['gamma_'+str(i)+'_'+str(j)],[force_vars['fenv_'+str(i)+'_'+str(j)][k+1]]])).evaluator().set_description(f"Velocity CC constraint for {j}-th point dim-{k} at time step {i}.")
                    prog.AddConstraint(partial(self.dummy_friction_residual_constraint,self.param['optimization_params']['mu_env']),lb=[0.],ub=[0.],vars=np.concatenate([force_vars['fenv_'+str(i)+'_'+str(j)],dummy_vars['dummy_'+str(i)+'_'+str(j)]]))
                    prog.AddConstraint(lambda z: [z[0]*z[1]],lb=[-np.inf],ub=[self.param['optimization_params']['comp_threshold']],vars=[dummy_vars['dummy_'+str(i)+'_'+str(j)],gamma_vars['gamma_'+str(i)+'_'+str(j)]])
                    prog.AddBoundingBoxConstraint(0,np.inf,gamma_vars['gamma_'+str(i)+'_'+str(j)]).evaluator().set_description(f"Gamma constraint for {j}-th point at {i}-th time step")

            # Constraints on manipulation contacts
            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                # Bounds on manipulation contact forces
                prog.AddBoundingBoxConstraint(0,10,force_vars['fmnp_'+str(i)+'_'+str(j)][0])
                for k in range(1,self.friction_dim+1):
                    prog.AddBoundingBoxConstraint(0,10,force_vars['fmnp_'+str(i)+'_'+str(j)][k]).evaluator().set_description(f"Bound for {j}-th mnp point at {i}-th time step")

                # Friction cone constraint on manipulation contacts
                prog.AddConstraint(partial(self.friction_cone_constraint,self.param['optimization_params']['mu_mnp']),lb=[0],ub=[np.inf],vars=force_vars['fmnp_'+str(i)+'_'+str(j)]).evaluator().set_description(f"Friction cone constraint for {j}-th mnp point at {i}-th time step")

                # Forece on and off
                if self.param['optimization_params']['force_on_off']:
                    if self.param['optimization_params']['use_alpha']:
                        prog.AddConstraint(partial(self.force_on_off,self.param['task_params']['manipulation_contact_points'][j]),lb=[4e-2,0],ub=[np.inf,0],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)+'_'+str(j)][0]],alpha_vars['alpha_'+str(i)+'_'+str(j)]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")               
                    else:
                        prog.AddConstraint(partial(self.force_on_off,self.param['task_params']['manipulation_contact_points'][j]),lb=[0.],ub=[self.param['optimization_params']['comp_threshold']],vars=np.concatenate([q[i],x[i],[force_vars['fmnp_'+str(i)+'_'+str(j)][0]]])).evaluator().set_description(f"Force on_off for {j}-th point at {i}-th time step")
                                
            # Balance constraints
            prog.AddConstraint(partial(self.force_balance_constraint,i),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],v[i]]+[force_vars['fenv_'+str(i)+'_'+str(j)] for j in range(len(self.index_set[i]))]+[force_vars['fmnp_'+str(i)+'_'+str(j)] for j in range(len(self.param['task_params']['manipulation_contact_points']))])).evaluator().set_description(f"Force balance at {i}-th time step")
            prog.AddConstraint(partial(self.torque_balance_constraint,i),lb=[0.,0.,0.],ub=[0.,0.,0.],vars=np.concatenate([q[i],x[i],w_axis[i],w_mag[i]]+[force_vars['fenv_'+str(i)+'_'+str(j)] for j in range(len(self.index_set[i]))]+[force_vars['fmnp_'+str(i)+'_'+str(j)] for j in range(len(self.param['task_params']['manipulation_contact_points']))])).evaluator().set_description(f"Torque balance at {i}-th time step")

        # Boundary constraints
        initial_pose_relaxation = self.param['optimization_params']['initial_pose_relaxation']
        goal_pose_relaxation = self.param['optimization_params']['goal_pose_relaxation']
        for i in range(4):
            prog.AddBoundingBoxConstraint(self.q_init[i]-initial_pose_relaxation, self.q_init[i]+initial_pose_relaxation, q[0][i]).evaluator().set_description(f"Initial orientation constraint {i}")
            prog.AddBoundingBoxConstraint(self.q_goal[i]-goal_pose_relaxation, self.q_goal[i]+goal_pose_relaxation, q[-1][i]).evaluator().set_description(f"Goal orientation constraint {i}")
        for i in range(3):
            prog.AddBoundingBoxConstraint(self.x_init[i]-initial_pose_relaxation, self.x_init[i]+initial_pose_relaxation, x[0][i]).evaluator().set_description(f"Initial position constraint {i}")
            prog.AddBoundingBoxConstraint(self.x_goal[i]-goal_pose_relaxation, self.x_goal[i]+goal_pose_relaxation, x[-1][i]).evaluator().set_description(f"Goal position constraint {i}")

        # Objective function
        prog.AddCost(self.cost_fun,vars=np.concatenate([q[i] for i in range(self.N)]+[x[i] for i in range(self.N)]))
        
        # Set up initial guess for the optimization
        for i in range(self.N):
            prog.SetInitialGuess(x[i], self.prev_sol['x'][i])
            prog.SetInitialGuess(q[i], self.prev_sol['q'][i])
            prog.SetInitialGuess(w_axis[i], self.prev_sol['w_axis'][i])
            prog.SetInitialGuess(w_mag[i], self.prev_sol['w_mag'][i])
            prog.SetInitialGuess(v[i], self.prev_sol['v'][i])

            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                prog.SetInitialGuess(force_vars['fmnp_'+str(i)+'_'+str(j)], self.prev_sol['fmnp_'+str(i)+'_'+str(j)])
                if self.param['optimization_params']['use_alpha']:
                    prog.SetInitialGuess(alpha_vars['alpha_'+str(i)+'_'+str(j)], self.prev_sol['alpha_'+str(i)+'_'+str(j)])

            for j in range(len(self.index_set[i])):
                if j < self.prev_sol['num_idx'][i]:
                    prog.SetInitialGuess(d_vars['d_'+str(i)+'_'+str(j)], self.prev_sol['d_'+str(i)+'_'+str(j)])
                    prog.SetInitialGuess(force_vars['fenv_'+str(i)+'_'+str(j)], self.prev_sol['fenv_'+str(i)+'_'+str(j)])
                    if self.param['optimization_params']['velocity_complementarity']:
                        prog.SetInitialGuess(gamma_vars['gamma_'+str(i)+'_'+str(j)], self.prev_sol['gamma_'+str(i)+'_'+str(j)])
                        prog.SetInitialGuess(dummy_vars['dummy_'+str(i)+'_'+str(j)], self.prev_sol['dummy_'+str(i)+'_'+str(j)])
                else:
                    prog.SetInitialGuess([force_vars['fenv_'+str(i)+'_'+str(j)][0]], [0])

        solver = SnoptSolver()

        snopt = SnoptSolver().solver_id()
        prog.SetSolverOption(snopt, "Major Iterations Limit", self.param['optimization_params']['max_mpcc_iter'])
        prog.SetSolverOption(snopt, "Major Feasibility Tolerance", self.param['optimization_params']['major_feasibility_tolerance'])
        prog.SetSolverOption(snopt, "Major Optimality Tolerance", self.param['optimization_params']['major_optimality_tolerance'])

        filename = f"tmp/debug.txt"
        prog.SetSolverOption(snopt, 'Print file', filename)
        result = solver.Solve(prog)

        res = {}

        res['num_idx'] = [len(sublist) for sublist in self.index_set]
        res['index_set'] = self.index_set
        res['is_success'] = result.is_success()

        res['q'] = result.GetSolution(q)
        res['x'] = result.GetSolution(x)
        res['w_axis'] = result.GetSolution(w_axis)
        res['w_mag'] = result.GetSolution(w_mag)
        res['v'] = result.GetSolution(v)
        res['dt'] = self.dt
        
        for i in range(self.N):
            for j in range(len(self.index_set[i])):
                res['fenv_'+str(i)+'_'+str(j)] = result.GetSolution(force_vars['fenv_'+str(i)+'_'+str(j)])
                res['d_'+str(i)+'_'+str(j)] = result.GetSolution(d_vars['d_'+str(i)+'_'+str(j)])
            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                res['fmnp_'+str(i)+'_'+str(j)] = result.GetSolution(force_vars['fmnp_'+str(i)+'_'+str(j)])

        if self.param['optimization_params']['velocity_complementarity']:
            for i in range(self.N):
                for j in range(len(self.index_set[i])):
                    res['dummy_'+str(i)+'_'+str(j)] = result.GetSolution(dummy_vars['dummy_'+str(i)+'_'+str(j)])
                    res['gamma_'+str(i)+'_'+str(j)] = result.GetSolution(gamma_vars['gamma_'+str(i)+'_'+str(j)])

        for i in range(self.N):
            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                res['fmnp_'+str(i)+'_'+str(j)] = result.GetSolution(force_vars['fmnp_'+str(i)+'_'+str(j)])
                if self.param['optimization_params']['use_alpha']:
                    res['alpha_'+str(i)+'_'+str(j)] = result.GetSolution(alpha_vars['alpha_'+str(i)+'_'+str(j)])
            for j in range(len(self.index_set[i])):
                res['fenv_'+str(i)+'_'+str(j)] = result.GetSolution(force_vars['fenv_'+str(i)+'_'+str(j)])

        if res['w_mag'].ndim == 1:
            res['w_mag'] = res['w_mag'][:, np.newaxis]

        return res