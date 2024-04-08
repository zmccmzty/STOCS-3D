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
import threading
from utils import *

class STOCS(object):

    def __init__(self, param):
        
        self.param = param
        self.N = self.param['optimization_params']['N']
        self.active_index_set = None

        # Task parameters
        self.q_init = self.param['task_params']['q_init']
        self.x_init = self.param['task_params']['x_init']
        self.q_goal = self.param['task_params']['q_goal']
        self.x_goal = self.param['task_params']['x_goal']

        self.manipuland = self.param['manipuland_params']['manipuland']

    def score_function(self,mpcc,res):
        q = res['q']
        x = res['x']
        N = self.param['optimization_params']['N']

        x_ = np.concatenate([q[i] for i in range(N)]+[x[i] for i in range(N)])
        score = mpcc.cost_fun(x_) + self.constraint_violation(mpcc,res)
        return score

    def line_search(self,mpcc,res_target,res_current,shrink_coef=0.8,max_iter_ls=20):
        current_score = self.score_function(mpcc,res_current)
        target_score = self.score_function(mpcc,res_target)

        iter_ls = 0
        step_size = 1
        res_temp = copy.deepcopy(res_target)
        while target_score > current_score and iter_ls < max_iter_ls:
            step_size = step_size*shrink_coef
            
            for i in range(self.N):
                res_temp['q'][i] = np.array(so3.quaternion(so3.interpolate(so3.from_quaternion(res_current['q'][i]),so3.from_quaternion(res_target['q'][i]),step_size)))
                res_temp['x'][i] = vectorops.interpolate(res_current['x'][i],res_target['x'][i],step_size)
                
                res_temp['w_mag'][i] = vectorops.interpolate(res_current['w_mag'][i],res_target['w_mag'][i],step_size)
                res_temp['w_axis'][i] = vectorops.interpolate(res_current['w_axis'][i],res_target['w_axis'][i],step_size)

                res_temp['v'][i] = vectorops.interpolate(res_current['v'][i],res_target['v'][i],step_size)

                if self.param['optimization_params']['velocity_complementarity']:
                    for j in range(len(self.active_index_set[i])):
                        res_temp['gamma_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['gamma_'+str(i)+'_'+str(j)],res_target['gamma_'+str(i)+'_'+str(j)],step_size)
                        res_temp['dummy_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['dummy_'+str(i)+'_'+str(j)],res_target['dummy_'+str(i)+'_'+str(j)],step_size)

                for j in range(len(self.active_index_set[i])):
                    res_temp['fenv_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['fenv_'+str(i)+'_'+str(j)],res_target['fenv_'+str(i)+'_'+str(j)],step_size)
                    res_temp['d_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['d_'+str(i)+'_'+str(j)],res_target['d_'+str(i)+'_'+str(j)],step_size)
                
                for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                    res_temp['fmnp_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['fmnp_'+str(i)+'_'+str(j)],res_target['fmnp_'+str(i)+'_'+str(j)],step_size)
                    if self.param['optimization_params']['use_alpha']:
                        res_target['alpha_'+str(i)+'_'+str(j)] = vectorops.interpolate(res_current['alpha_'+str(i)+'_'+str(j)],res_target['alpha_'+str(i)+'_'+str(j)],step_size)
            
            target_score = self.score_function(mpcc,res_temp)

            iter_ls += 1

        return res_temp

    def oracle(self,q,x,index_set=None):

        add_threshold = self.param['optimization_params']['add_threshold']
        disturbances = self.param['optimization_params']['disturbances']
        time_smoothing_step = self.param['optimization_params']['time_smoothing_step']

        if index_set is None:
            index_set = [[] for _ in range(self.N)]
        else:
            index_set = copy.deepcopy(index_set)

        len_in = len([index_point for sub_index_set in index_set for index_point in sub_index_set])

        closest_points = [[] for _ in range(self.N)]
        for ti, (qi,xi) in enumerate(zip(q,x)):
            
            closest_points_ti = closest_points[ti]

            q_unit = vectorops.unit(qi)
            self.manipuland.setTransform((so3.from_quaternion(q_unit),xi))

            for env_idx, environment in enumerate(self.param['environment_params']['environments']):
                dist_, p_obj, p_env = self.manipuland.distance(environment)
                p_obj_local = se3.apply(se3.inv((so3.from_quaternion(q_unit),xi)),p_obj)

                if dist_< self.param['optimization_params']['active_threshold']: 
                    closest_points_ti.append(p_obj_local+[env_idx])

                # Spatial Smoothing
                for disturbance in disturbances:
                    for idx in range(3):
                        if idx == 0:
                            q_p = so3.quaternion(so3.mul(so3.from_axis_angle(([1,0,0],disturbance)),so3.from_quaternion(q_unit)))
                            q_n = so3.quaternion(so3.mul(so3.from_axis_angle(([1,0,0],-disturbance)),so3.from_quaternion(q_unit)))
                        elif idx == 1:
                            q_p = so3.quaternion(so3.mul(so3.from_axis_angle(([0,1,0],disturbance)),so3.from_quaternion(q_unit)))
                            q_n = so3.quaternion(so3.mul(so3.from_axis_angle(([0,1,0],-disturbance)),so3.from_quaternion(q_unit)))
                        elif idx == 2:
                            q_p = so3.quaternion(so3.mul(so3.from_axis_angle(([0,0,1],disturbance)),so3.from_quaternion(q_unit)))
                            q_n = so3.quaternion(so3.mul(so3.from_axis_angle(([0,0,1],-disturbance)),so3.from_quaternion(q_unit)))

                        for q_ in [q_p,q_n]:
                            self.manipuland.setTransform((so3.from_quaternion(q_),xi))
                            dist_, p_obj, p_env = self.manipuland.distance(environment)
                            p_obj_local = se3.apply(se3.inv((so3.from_quaternion(q_),xi)),p_obj)

                            if dist_< 0.01:     
                                closest_points_ti.append(p_obj_local+[env_idx])

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
                            self.manipuland.setTransform((so3.from_quaternion(q_unit),x_))
                            dist_, p_obj, p_env = self.manipuland.distance(environment)
                            p_obj_local = se3.apply(se3.inv((so3.from_quaternion(q_unit),x_)),p_obj)
                            
                            if dist_< self.param['optimization_params']['active_threshold']:
                                closest_points_ti.append(p_obj_local+[env_idx])
            
        # Temporal Smoothing
        index_set_ = [[] for _ in range(self.N)]
        for ti in range(self.N):
            index_set_[ti] = copy.deepcopy(index_set[ti]) 
            for t_ in range(ti-time_smoothing_step,ti+time_smoothing_step+1):
                if t_ < 0:
                    pass
                elif t_ > self.N-1:
                    pass
                else: 
                    for point in closest_points[t_]:
                        if point not in index_set_[ti]:
                            dist_min = np.inf
                            for i in range(len(index_set_[ti])):
                                dist = vectorops.norm(vectorops.sub(point,index_set_[ti][i]))
                                if dist < dist_min:
                                    dist_min = dist
                            if dist_min > add_threshold:
                                index_set_[ti].append(point)

        index_set = copy.deepcopy(index_set_)

        len_out = len([index_point for sub_index_set in index_set for index_point in sub_index_set])

        # Include all points in the point cloud. Used for testing pure MPCC.
        # if index_set is None:
        #     index_set = [[] for _ in range(self.N)]
        #     for _ in index_set:
        #         for i in range(len(self.object_pc)):
        #             _.append(self.object_pc[i])
        # else:
        #     index_set = copy.deepcopy(index_set)

        return index_set

    def plot_index_points(self,q,x):
        xyz = np.zeros((len(self.active_index_set[0]),3))
        for i,index_pt in enumerate(self.active_index_set[0]):
            xyz[i,:] = index_pt
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        coord.scale(0.1, center=[0,0,0])
        T = se3.ndarray((so3.from_quaternion(q),x))
        self.param['manipuland_params']['manipuland_o3d'].transform(T)
        pcd.transform(T)
        coord.translate(x)
        #o3d.visualization.draw_geometries([object_mesh_o3d,pcd,coord])
        o3d.visualization.draw_geometries([self.param['manipuland_params']['manipuland_o3d'],pcd])
        self.param['manipuland_params']['manipuland_o3d'].transform(np.linalg.inv(T))
        return None 

    def constraint_violation(self,mpcc,res):
        score = self.index_point_penetration(res) + self.deepest_penetration(res) + \
                self.complimentarity_residual(mpcc,res) + self.balance_residual(mpcc,res) +\
                self.unit_constraint_violation(res) + self.boundary_constraint_violation(res) +\
                self.integration_constraint_violation(mpcc,res)
        return score

    def unit_constraint_violation(self,res):
        violation = 0
        for i in range(self.N):
            q = res['q'][i]
            w_axis = res['w_axis'][i]
            violation += np.abs(q@q-1)
            violation += np.abs(w_axis@w_axis-1)
        
        return violation

    def boundary_constraint_violation(self,res):
        violation = 0
        for i in range(self.N):
            violation += max(0,res['x'][i][0]-self.param['task_params']['x_bound'][1][0]) + max(0,self.param['task_params']['x_bound'][0][0]-res['x'][i][0])
            violation += max(0,res['x'][i][1]-self.param['task_params']['x_bound'][1][1]) + max(0,self.param['task_params']['x_bound'][0][1]-res['x'][i][1])
            violation += max(0,res['x'][i][2]-self.param['task_params']['x_bound'][1][2]) + max(0,self.param['task_params']['x_bound'][0][2]-res['x'][i][2])

            violation += max(0,res['v'][i][0]-self.param['task_params']['v_bound'][1][0]) + max(0,self.param['task_params']['v_bound'][0][0]-res['v'][i][0])
            violation += max(0,res['v'][i][1]-self.param['task_params']['v_bound'][1][1]) + max(0,self.param['task_params']['v_bound'][0][1]-res['v'][i][1])
            violation += max(0,res['v'][i][2]-self.param['task_params']['v_bound'][1][2]) + max(0,self.param['task_params']['v_bound'][0][2]-res['v'][i][2])

            lb = -np.pi
            ub = np.pi
            violation += max(0,res['w_mag'][i]-ub) + max(0,lb-res['w_mag'][i])


            for j in range(len(self.active_index_set[i])):
                lb = 0
                ub = 10
                for k in range(self.param['optimization_params']['friction_dim']+1):
                    violation += max(0,res['fenv_'+str(i)+'_'+str(j)][k]-ub) + max(0,lb-res['fenv_'+str(i)+'_'+str(j)][k])

                lb = 0
                ub = np.inf
                violation += max(0,res['d_'+str(i)+'_'+str(j)][0]-ub) + max(0,lb-res['d_'+str(i)+'_'+str(j)][0])

            for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                lb = 0
                ub = 10
                violation += max(0,res['fmnp_'+str(i)+'_'+str(j)][k]-ub) + max(0,lb-res['fmnp_'+str(i)+'_'+str(j)][k])

            if self.param['optimization_params']['velocity_complementarity']:
                for j in range(len(self.active_index_set[i])):
                    lb = 0
                    ub = np.inf
                    violation += max(0,res['gamma_'+str(i)+'_'+str(j)][0]-ub) + max(0,lb-res['gamma_'+str(i)+'_'+str(j)][0])
                    violation += max(0,res['dummy_'+str(i)+'_'+str(j)][0]-ub) + max(0,lb-res['dummy_'+str(i)+'_'+str(j)][0])

        return violation

    def integration_constraint_violation(self,mpcc,res):
        violation = 0
        for i in range(1,self.N):
            violation += np.sum(np.abs(mpcc.backward_euler_q(np.concatenate((res['q'][i],res['q'][i-1],res['w_axis'][i],res['w_mag'][i])),res['dt'][i])))
            violation += np.sum(np.abs(mpcc.backward_euler_x(np.concatenate((res['x'][i],res['x'][i-1],res['v'][i])),res['dt'][i])))

        return violation

    def index_point_penetration(self,res):
        penetration = 0
        for i in range(self.N):
            qi = res['q'][i]
            xi = res['x'][i]
            for point in self.active_index_set[i]:
                point_world = se3.apply((so3.from_quaternion(qi),xi),point)   
                for environment in self.param['environment_params']['environments']:
                    dist = environment.distance(point_world)[0]
                    penetration += abs(min(dist,0))
        return penetration

    def deepest_penetration(self,res):

        penetration = 0
        for i in range(self.N):
            qi = res['q'][i]
            xi = res['x'][i]
            self.manipuland.setTransform((so3.from_quaternion(qi),xi))
            for environment in self.param['environment_params']['environments']:
                dist_, p_obj, p_env = self.manipuland.distance(environment)
                if dist_ < 0:
                    penetration += abs(min(dist_,0))

        return penetration

    def complimentarity_residual(self,mpcc,res):

        q = res['q']
        x = res['x']
        w_axis = res['w_axis']
        w_mag = res['w_mag']
        v = res['v']

        residual = 0

        for i in range(self.N):
            for j in range(len(self.active_index_set[i])):
                # Position Comp
                residual += abs(min(self.param['optimization_params']['comp_threshold']-res['fenv_'+str(i)+'_'+str(j)][0]*res['d_'+str(i)+'_'+str(j)][0],0))
                
                # Velocity Comp
                if self.param['optimization_params']['velocity_complementarity']:
                    for k in range(self.param['optimization_params']['friction_dim']):
                        x_ = np.concatenate([q[i],w_axis[i],w_mag[i],x[i],v[i],res['gamma_'+str(i)+'_'+str(j)],[res['fenv_'+str(i)+'_'+str(j)][k+1]]])
                        constraint_value = partial(mpcc.velocity_cc_constraint,self.active_index_set[i][j],k)(x_)
                        residual += abs(min(self.param['optimization_params']['comp_threshold'] - constraint_value[0],0))                   
                    residual += abs(min(self.param['optimization_params']['comp_threshold'] - res['dummy_'+str(i)+'_'+str(j)][0]*res['gamma_'+str(i)+'_'+str(j)][0],0))

        return residual

    def balance_residual(self,mpcc,res):

        q = res['q']
        x = res['x']
        v = res['v']
        w_axis = res['w_axis']
        w_mag = res['w_mag']

        residual = 0
        for i in range(self.N):
            x_ = np.concatenate([q[i],x[i],v[i]]+[res['fenv_'+str(i)+'_'+str(j)] for j in range(len(self.active_index_set[i]))]+[res['fmnp_'+str(i)+'_'+str(j)] for j in range(len(self.param['task_params']['manipulation_contact_points']))])
            residual += np.sum(np.abs(mpcc.force_balance_constraint(i,x_)))
            x_ = np.concatenate([q[i],x[i],w_axis[i],w_mag[i]]+[res['fenv_'+str(i)+'_'+str(j)] for j in range(len(self.active_index_set[i]))]+[res['fmnp_'+str(i)+'_'+str(j)] for j in range(len(self.param['task_params']['manipulation_contact_points']))])
            residual += np.sum(np.abs(mpcc.torque_balance_constraint(i,x_))) 

        return residual

    def plot_index_points(self,q,x):
        xyz = np.zeros((len(self.active_index_set[0]),3))
        for i,index_pt in enumerate(self.active_index_set[0]):
            xyz[i,:] = index_pt[:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        coord.scale(0.1, center=[0,0,0])
        T = se3.ndarray((so3.from_quaternion(q),x))
        self.param['manipuland_params']['manipuland_o3d'].transform(T)
        pcd.transform(T)
        coord.translate(x)
        o3d.visualization.draw_geometries([self.param['manipuland_params']['manipuland_o3d'],pcd])
        self.param['manipuland_params']['manipuland_o3d'].transform(np.linalg.inv(T))
        return None 

    def solve(self):
        
        t_start = time.time()
        iter = 0
        stocs_res = {}
        stocs_res['param'] = self.param
        stocs_res['is_success'] = False
        index_set_nums = []

        while iter < self.param['optimization_params']['stocs_max_iter']:
            
            if iter == 0:
                if self.param['optimization_params']['initialization'] == "initial":
                    q = np.tile(np.array(self.q_init),(self.N,1))
                    x = np.tile(np.array(self.x_init),(self.N,1))
                elif self.param['optimization_params']['initialization'] == "linear":
                    q = np.zeros((self.N,4))
                    for i in range(self.N):
                        q[i,:] = so3.quaternion(so3.interpolate(so3.from_quaternion(self.q_init),so3.from_quaternion(self.q_goal),i/(self.N-1)))
                    x = np.zeros((self.N,3))
                    for i in range(self.N):
                        x[i,:] =vectorops.interpolate(self.x_init,self.x_goal,i/(self.N-1))        
            else:
                q = prev_sol['q']
                x = prev_sol['x']

            self.active_index_set = self.oracle(q,x,self.active_index_set)
            index_set_nums.append([len(_) for _ in self.active_index_set])
            if iter == 0:
                error = so3.error(so3.from_quaternion(self.param['task_params']['q_goal']),so3.from_quaternion(self.param['task_params']['q_init']))
                distance = vectorops.norm(error)
                axis = vectorops.unit(error)
                if self.param['optimization_params']['initialization'] == "initial":
                    w_axis = np.tile(np.array([1,0,0]),(self.N,1))
                    w_mag = np.tile(np.array([0.0]),(self.N,1))
                    v = np.tile(np.array([0.,0.,0.]),(self.N,1))
                elif self.param['optimization_params']['initialization'] == "linear":
                    w_axis = np.tile(np.array([1,0,0]),(self.N,1))
                    w_mag = np.ones((self.N,1))*(distance/self.N)
                    v = np.tile((np.array(self.x_goal)-np.array(self.x_init))/(self.N*self.param['optimization_params']['time_step']),(self.N,1))

                prev_sol = {}
                
                prev_sol['num_idx'] = [len(sublist) for sublist in self.active_index_set]
                prev_sol['index_set'] = self.active_index_set

                prev_sol['q'] = q
                prev_sol['x'] = x
                prev_sol['w_axis'] = w_axis
                prev_sol['w_mag'] = w_mag
                prev_sol['v'] = v
                prev_sol['dt'] = [0.0]+[self.param['optimization_params']['time_step']]*(self.N-1) 

                if self.param['optimization_params']['velocity_complementarity']:
                    for i in range(self.N):
                        for j in range(len(self.active_index_set[i])):
                            prev_sol['gamma_'+str(i)+'_'+str(j)] = [0.0]
                            prev_sol['dummy_'+str(i)+'_'+str(j)] = [0.0]

                for i in range(self.N):
                    for j in range(len(self.active_index_set[i])):
                        prev_sol['fenv_'+str(i)+'_'+str(j)] = [1e-3] + [1e-3]*self.param['optimization_params']['friction_dim']
                        prev_sol['d_'+str(i)+'_'+str(j)] = [0.0]
                    for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                        prev_sol['fmnp_'+str(i)+'_'+str(j)] = [1e-3] + [1e-3]*self.param['optimization_params']['friction_dim']
                        if self.param['optimization_params']['use_alpha']:
                            prev_sol['alpha_'+str(i)+'_'+str(j)] = [0.0]

            elif [len(sublist) for sublist in prev_sol['index_set']] != [len(sublist) for sublist in self.active_index_set]:
                for i in range(self.N):
                    if len(self.active_index_set[i]) != len(prev_sol['index_set'][i]):
                        for j in range(len(prev_sol['index_set'][i]),len(self.active_index_set[i])):
                            prev_sol['d_'+str(i)+'_'+str(j)] = [0.0]
                            prev_sol['fenv_'+str(i)+'_'+str(j)] = [1e-3] + [1e-3]*self.param['optimization_params']['friction_dim']

                            if self.param['optimization_params']['velocity_complementarity']:
                                prev_sol['gamma_'+str(i)+'_'+str(j)] = [0.0]
                                prev_sol['dummy_'+str(i)+'_'+str(j)] = [0.0]

            print(f"Index points for each time step along the trajectory: {[len(sublist) for sublist in self.active_index_set]}")

            self.param['optimization_params']['max_mpcc_iter'] = min(5+5*iter,100)
            mpcc = MPCC(self.active_index_set,self.param,prev_sol)
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
                res = self.line_search(mpcc,res_target,res_current=prev_sol)
            finally:
                stop_event.set()
                dot_thread.join()
                print()
        
            q = []
            q_prev = []
            x = []
            x_prev = []
            w_axis = []
            w_axis_prev = []
            w_mag = []
            w_mag_prev = []
            v = []
            v_prev = []
            f_env_res = []
            f_mnp_res = []
            f_env_prev = []
            f_mnp_prev = []
            for i in range(self.N):
                q.append(res['q'][i])
                q_prev.append(prev_sol['q'][i])
                x.append(res['x'][i])
                x_prev.append(prev_sol['x'][i])
                w_axis.append(res['w_axis'][i])
                w_axis_prev.append(prev_sol['w_axis'][i])
                w_mag.append(res['w_mag'][i])
                w_mag_prev.append(prev_sol['w_mag'][i])
                v.append(res['v'][i])
                v_prev.append(prev_sol['v'][i])
                for j in range(len(self.active_index_set[i])):
                    f_env_res.append(res['fenv_'+str(i)+'_'+str(j)])
                    f_env_prev.append(prev_sol['fenv_'+str(i)+'_'+str(j)])
                for j in range(len(self.param['task_params']['manipulation_contact_points'])):
                    f_mnp_res.append(res['fmnp_'+str(i)+'_'+str(j)])
                    f_mnp_prev.append(prev_sol['fmnp_'+str(i)+'_'+str(j)])
            
            q = np.array(q).reshape((-1,1))
            q_prev = np.array(q_prev).reshape((-1,1))
            x = np.array(x).reshape((-1,1))
            x_prev = np.array(x_prev).reshape((-1,1))
            w_axis = np.array(w_axis).reshape((-1,1))
            w_axis_prev = np.array(w_axis_prev).reshape((-1,1))
            w_mag = np.array(w_mag).reshape((-1,1))
            w_mag_prev = np.array(w_mag_prev).reshape((-1,1))
            v = np.array(v).reshape((-1,1))
            v_prev = np.array(v_prev).reshape((-1,1))
            f_env = np.array(f_env_res).reshape((-1,1))
            f_mnp = np.array(f_mnp_res).reshape((-1,1))
            f_env_prev = np.array(f_env_prev).reshape((-1,1))
            f_mnp_prev = np.array(f_mnp_prev).reshape((-1,1))

            q_diff = q - q_prev
            x_diff = x - x_prev
            w_axis_diff = w_axis - w_axis_prev
            w_mag_diff = w_mag - w_mag_prev
            v_diff = v - v_prev
            f_env_diff = f_env - f_env_prev
            f_mnp_diff = f_mnp - f_mnp_prev

            var_diff = np.concatenate([q_diff,x_diff,w_axis_diff,w_mag_diff,v_diff,f_env_diff,f_mnp_diff])
            
            if self.complimentarity_residual(mpcc,res) < 1e-4*(1+1+self.param['optimization_params']['friction_dim'])*sum(len(sublist) for sublist in self.active_index_set) and self.deepest_penetration(res) < 1e-4*self.N and self.balance_residual(mpcc,res) < 1e-4*self.N and np.linalg.norm(var_diff) < 1e-4*len(var_diff):
                
                print("Successfully found result.")
                iter += 1
                stocs_res[f'iter_{iter}'] = res
                stocs_res['is_success'] = True
                break
            
            iter += 1
            prev_sol = res.copy()
            stocs_res[f'iter_{iter}'] = res
        
        stocs_res['total_iter'] = iter
        print(f"Total iterations: {iter}.")  
        print(f"Solve time: {time.time()-t_start}s.")
        print(f"Average index point: {np.mean(np.array(index_set_nums))}")        
        return stocs_res,np.mean(np.array(index_set_nums))

if __name__ == '__main__':

    # Take arguments from the command line
    import argparse
    import importlib
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=str, default='plane', help='Environment')
    parser.add_argument('--manipuland', type=str, default='mustard', help='Manipuland')

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
    # Access the 'params' attribute from the imported module
    params = getattr(module, 'params', None)

    if params is None:
        print(f"The 'params' attribute was not found in the module {module_name}")
    else:
        print(f"Successfully imported 'params' from {module_name}")
        print(f"Environment: {environment}")
        print(f"Manipuland: {manipuland}")
        print(f"Number of time steps: {params['optimization_params']['N']}")
        print(f"Initialization: {params['optimization_params']['initialization']}")
        print(f"Velocity complementarity: {params['optimization_params']['velocity_complementarity']}")
        print(f"Assumption: {params['optimization_params']['assumption']}")
        print(f"Force on off: {params['optimization_params']['force_on_off']}")
        print(f"Use alpha: {params['optimization_params']['use_alpha']}")

    params['optimization_params']['time_smoothing_step'] = 1
    params['optimization_params']['disturbances'] = [1e-2]

    stocs_3d = STOCS(params)

    t_start = time.time()
    res,average_index_point = stocs_3d.solve()
    t_used = time.time() - t_start
    print(f"Time used: {t_used}s.")
    print(f"Average index point: {average_index_point}")
    SF = res['is_success']
    experiment_result = {"Object": manipuland, "S/F": SF, "Iterations": res['total_iter'], "Time": t_used, "Average Contacts": average_index_point}

    # Save the result
    environment = params['environment_params']['environment_name']
    if not os.path.exists(f'results/{environment}_{manipuland}_STOCS'):
        os.makedirs(f'results/{environment}_{manipuland}_STOCS')
    res['param']['manipuland_params']['manipuland'] = None
    res['param']['manipuland_params']['manipuland_g3d'] = None
    res['param']['manipuland_params']['manipuland_o3d'] = None
    res['param']['environment_params']['environments'] = None
    res['param']['environment_params']['query_object_ad'] = None
    res['param']['environment_params']['query_object_f'] = None
    res['param']['environment_params']['query_object_ad_ids'] = None
    res['param']['environment_params']['query_object_f_ids'] = None
    np.save(f'results/{environment}_{manipuland}_STOCS/result.npy', res)