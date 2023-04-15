from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import multiprocessing as mp
from joblib import delayed,Parallel
import ctypes
import os
import json
import warnings
from itertools import combinations

    
def random_se3(mags=np.ones(6)):
    Extran = np.eye(4)
    Extran[:3,:3] = vec_tran(np.random.rand(3)*mags[:3])
    Extran[:3,3] = np.random.rand(3)*mags[3:]
    return Extran

def random_so3(mags=np.ones(3)):
    Extran = vec_tran(np.random.rand(3)*mags)
    return Extran

def fake_edges(num_edges,mag=np.ones(3)):
    Extran = np.eye(4)
    Extran[:3,:3] = vec_tran(mag*np.random.rand(3))
    Extran[:3,3] = np.random.rand(3)
    scale = 3.5 + 0.5*np.random.rand()
    pcd_edge = [random_se3() for _ in range(num_edges)]
    camera_edge = []
    for edge in pcd_edge:
        cedge = np.linalg.inv(Extran) @ edge @ Extran
        cedge[:3,3] /= scale
        camera_edge.append(cedge)
    return pcd_edge,camera_edge,Extran,scale

def fake_rotedges(num_edges,mag=np.ones(3)):
    Extran = vec_tran(np.random.rand(3))
    pcd_edge = [random_so3(mag) for _ in range(num_edges)]
    camera_edge = []
    for edge in pcd_edge:
        cedge = Extran.T @ edge @ Extran
        camera_edge.append(cedge)
    return pcd_edge,camera_edge,Extran   

def read_camera_json(json_path:str,toF0=True):
    with open(json_path,'r')as f:
        data = json.load(f)
    Pose = []
    for extran in data['extrinsics']:
        T = np.eye(4)
        R = np.array(extran['value']['rotation'])
        t = np.array(extran['value']['center'])
        T[:3,:3] = R
        # T[:3,3] = t
        T[:3,3] = -R.dot(t)
        Pose.append(T)
    InvPose = [np.linalg.inv(T) for T in Pose]  # world to self format -> self to world format
    if toF0:
        pose0 = Pose[0]
        InvPose = [pose0 @ pose for pose in InvPose]
    return InvPose

def read_pcd_json(json_path:str):
    o3dPoseGraph = o3d.io.read_pose_graph(json_path)
    Pose = []
    for node in o3dPoseGraph.nodes:
        Pose.append(node.pose)
    inv_t0 = np.linalg.inv(Pose[0])
    Pose = [inv_t0 @ T for T in Pose]
    return Pose

def read_pose(filename:str):
    with open(filename,'r')as f:
        data = [line.rstrip('\n') for line in f.readlines()]
    if len(data) == 4:
        T = np.loadtxt(filename,dtype=np.float64)
    elif len(data) == 2:
        translation = np.fromstring(data[0],dtype=np.float32,sep=',')
        quat = np.fromstring(data[1],dtype=np.float32,sep=',')
        T = np.eye(4)
        R = Rotation.from_quat(quat)
        T[:3,:3] = R.as_matrix()
        T[:3,3] = translation
    else:
        raise RuntimeError("Unknown pose type.")
    return T

def read_pcd_pose(pose_dir:str,toF0=True):
    
    pose_files = list(sorted(os.listdir(pose_dir)))
    Pose = []
    for pose_filename in pose_files:
        Pose.append(read_pose(os.path.join(pose_dir,pose_filename)))
    if toF0:
        inv_t0 = np.linalg.inv(Pose[0])
        Pose = [inv_t0 @ T for T in Pose]
    return Pose

def read_pcd_se3(pose_dir:str):
    pose_files = list(sorted(os.listdir(pose_dir)))
    Pose = []
    for pose_filename in pose_files:
        Pose.append(np.loadtxt(os.path.join(pose_dir,pose_filename),dtype=np.float32,delimiter=' '))
    return Pose

def poseToAdjedge(Pose:list):
    Edge = list()
    for i in range(len(Pose)-1):
        Edge.append(Pose[i+1] @ np.linalg.inv(Pose[i]))
    return Edge

def poseToFulledge(Pose:list):
    Edge = list()
    inv_pose = [np.linalg.inv(pose) for pose in Pose]
    for idx_k, idx_kp in combinations(range(len(Pose)),2):
        Edge.append(Pose[idx_kp] @ inv_pose[idx_k])
    return Edge

def fullEdge_Idx(Pose:list):
    Edge = list()
    inv_pose = [np.linalg.inv(pose) for pose in Pose]
    idx_list = []
    for idx_k, idx_kp in combinations(range(len(Pose)),2):
        Edge.append(Pose[idx_kp] @ inv_pose[idx_k])
        idx_list.append([idx_k,idx_kp])
    return Edge, idx_list

def nptrans(pcd:np.ndarray,G:np.ndarray)->np.ndarray:
    R,t = G[:3,:3], G[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t

def euler_tran(x:np.ndarray,degrees=True):
    R = Rotation.from_euler('zyx',x[:3],degrees=degrees)
    t = x[3:]
    Tr = np.eye(4)
    Tr[:3,:3] = R.as_matrix()
    Tr[:3,3] = t
    return Tr

def vec_tran(rot_vec:np.ndarray):
    R = Rotation.from_rotvec(rot_vec)
    return R.as_matrix()

def toVec(Rmat:np.ndarray):
    R = Rotation.from_matrix(Rmat)
    vecR = R.as_rotvec()
    return vecR

def TL_solve(camera_edge,pcd_edge):
    assert len(camera_edge) == len(pcd_edge)
    N = len(camera_edge)
    alpha = np.zeros([N,3])
    beta = alpha.copy()
    for i,(cedge,pedge) in enumerate(zip(camera_edge,pcd_edge)):
        cvec = toVec(cedge)
        pvec = toVec(pedge)
        alpha[i,:] = cvec
        beta[i,:] = pvec
    H = np.dot(beta.T,alpha)  # (3,3)
    U, S, Vt = np.linalg.svd(H)  
    R = np.dot(Vt.T, U.T)     
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    return R, S, alpha, beta



def FGR_TL_solve(camera_edge,pcd_edge):
    assert len(camera_edge) == len(pcd_edge)
    N = len(camera_edge)
    alpha = np.zeros([N,3])
    beta = alpha.copy()
    for i,(cedge,pedge) in enumerate(zip(camera_edge,pcd_edge)):
        cvec = toVec(cedge)
        pvec = toVec(pedge)
        alpha[i,:] = cvec
        beta[i,:] = pvec
    A = o3d.geometry.PointCloud()
    B = o3d.geometry.PointCloud()
    A.points = o3d.utility.Vector3dVector(alpha)
    B.points = o3d.utility.Vector3dVector(beta)
    idx = np.repeat(np.arange(N)[:,None],repeats=2,axis=1)
    corres = o3d.utility.Vector2iVector(idx)
    res = o3d.pipelines.registration.registration_fgr_based_on_correspondence(
            B, A, corres,o3d.pipelines.registration.FastGlobalRegistrationOption(
                decrease_mu=True,use_absolute_scale=True,division_factor=2
            ))
    return res.transformation[:3,:3]

class EdgeConsistencyLoss:
    def __init__(self,camera_edge,pcd_edge):
        assert len(camera_edge) == len(pcd_edge), "camera edge (%d) != pcd edge (%d)"%(len(camera_edge),len(pcd_edge))
        self.edge_num = len(camera_edge)
        self.camera_edge = camera_edge
        self.pcd_edge = pcd_edge
        
    def transform_loss(self,se3_tran:np.ndarray,scale):
        loss_list = np.zeros([self.edge_num,6])
        for i in range(self.edge_num):
            camera_tran = self.camera_edge[i].copy()
            camera_tran[:3,3] *= scale
            error_se3 = np.linalg.inv(self.pcd_edge[i] @ se3_tran) @ se3_tran @ camera_tran
            loss_list[i,:3] = toVec(error_se3[:3,:3])
            loss_list[i,3:] = error_se3[:3,3]
        return loss_list.mean(axis=0)
    
    def direct_loss(self,twist_by_scale:np.ndarray):
        scale = twist_by_scale[-1]
        loss_list = np.zeros([self.edge_num,6])
        se3_tran = np.eye(4)
        se3_tran[:3,:3] = vec_tran(twist_by_scale[:3])
        se3_tran[:3,3] = twist_by_scale[3:-1]
        for i in range(self.edge_num):
            camera_tran = self.camera_edge[i].copy()
            camera_tran[:3,3] *= scale
            error_se3 = np.linalg.inv(self.pcd_edge[i] @ se3_tran) @ se3_tran @ camera_tran
            loss_list[i,:3] = toVec(error_se3[:3,:3])
            loss_list[i,3:] = error_se3[:3,3]
        return loss_list.mean()
    
    def consistency_loss(self,tran:np.ndarray):
        scales = np.zeros(self.edge_num)
 
        for i in range(self.edge_num):
            t0 = self.pcd_edge[i][:3,3] - (np.eye(3)-self.pcd_edge[i][:3,:3]) @ tran
            t1 = self.camera_edge[i][:3,3]
            t0_norm, t1_norm = map(np.linalg.norm,[t0,t1])
            scales[i] = t0_norm/t1_norm

        return np.var(scales)
    
    def procfun_consistency_loss(self,X:np.ndarray,Y:mp.Array,
                                       lock:mp.Lock,count:mp.Value,max_num:int,batch_num:int):
        while True:
            with lock:
                local_cnt = count.value
                count.value += batch_num
            if local_cnt >= max_num:
                break
            end_index = min(local_cnt+batch_num,max_num)
            for idx in range(local_cnt,end_index):
                x = X[idx]
                loss = self.consistency_loss(x)
                Y[idx] = loss.item()

    def proc_consistency_loss(self,X:np.ndarray,proc_num:int,batch_num:int):
        proc_num = os.cpu_count() if proc_num <0 else proc_num
        Loss = mp.Array(ctypes.c_double,X.shape[0])
        count = mp.Value(ctypes.c_int)
        lock = mp.Lock()
        proc_list = [mp.Process(target=self.procfun_consistency_loss,
                                args=(X,Loss,lock,count,X.shape[0],batch_num)) for _ in range(proc_num)]
        [proc.start() for proc in proc_list]
        [proc.join() for proc in proc_list]
        return np.array(Loss[:],dtype=ctypes.c_double)

    def solve_Scale(self,tran:np.ndarray):
        scales = np.zeros(self.edge_num)
        for i in range(self.edge_num):
            t0 = self.pcd_edge[i][:3,3] - (np.eye(3)-self.pcd_edge[i][:3,:3]) @ tran
            t1 = self.camera_edge[i][:3,3]
            scales[i] = np.linalg.norm(t0)/np.linalg.norm(t1)
        return np.mean(scales)

    def solve_R(self,tran:np.ndarray,scale) -> np.ndarray:
        P0 = np.zeros([self.edge_num,3])
        P1 = np.zeros([self.edge_num,3])
        for i in range(self.edge_num):
            t0 = self.pcd_edge[i][:3,3] - (np.eye(3)-self.pcd_edge[i][:3,:3]) @ tran
            t1 = self.camera_edge[i][:3,3]*scale
            P0[i] = t0
            P1[i] = t1
        R = self.SVD_Rotation(P1,P0)  # p0 = Rp1
        return R
    
    @staticmethod
    def SVD_Rotation(src:np.ndarray,dst:np.ndarray) -> np.ndarray:
        m = src.shape[1]
        # translate points to their centroids
        centroid_A = np.mean(src, axis=0)
        centroid_B = np.mean(dst, axis=0)
        AA = src - centroid_A
        BB = dst - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        return R


class EdgeLoss:
    def __init__(self,camera_edge,pcd_edge,init_rot:np.eye(3),noise_bnd=0.1):
        assert len(camera_edge) == len(pcd_edge), "camera edge (%d) != pcd edge (%d)"%(len(camera_edge),len(pcd_edge))
        self.edge_num = len(camera_edge)
        self.camera_edge = camera_edge
        self.pcd_edge = pcd_edge
        self.init_rot = init_rot
        self.noise_bnd = noise_bnd
    def transform_loss(self,TCL:np.ndarray,scale):
        loss_list = np.zeros([self.edge_num,6])
        for i in range(self.edge_num):
            camera_tran = self.camera_edge[i].copy()
            camera_tran[:3,3] *= scale
            error_se3 = np.linalg.inv(TCL @ self.pcd_edge[i]) @ camera_tran @ TCL
            loss_list[i,:3] = toVec(error_se3[:3,:3])
            loss_list[i,3:] = error_se3[:3,3]
        return loss_list.mean(axis=0)
    
    def rotation_loss(self,rot_vec:np.ndarray,reduction='mean'):
        loss = np.zeros(self.edge_num)
        RCL = vec_tran(rot_vec).dot(self.init_rot)
        for i,(cedge,pedge) in enumerate(zip(self.camera_edge,self.pcd_edge)):
            RC = cedge[:3,:3]
            RL = pedge[:3,:3]
            Rerr = RC @ RCL - RCL @ RL 
            loss[i] = np.sqrt(np.sum(Rerr**2))
            # loss[i] = min(np.sqrt(np.sum(Rerr**2)),self.noise_bnd)
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
    
    def procfun_rotation_loss(self,X:np.ndarray,Y:mp.Array,
                                lock:mp.Lock,count:mp.Value,max_num:int,batch_num:int):
        while True:
            with lock:
                local_cnt = count.value
                count.value += batch_num
            if local_cnt >= max_num:
                break
            end_index = min(local_cnt+batch_num,max_num)
            for idx in range(local_cnt,end_index):
                x = X[idx]
                loss = self.rotation_loss(x)
                Y[idx] = loss
    
    def proc_rotation_loss(self,X:np.ndarray,proc_num:int,batch_num:int):
        proc_num = os.cpu_count() if proc_num <0 else proc_num
        Loss = mp.Array(ctypes.c_double,X.shape[0])
        count = mp.Value(ctypes.c_int)
        lock = mp.Lock()
        proc_list = [mp.Process(target=self.procfun_rotation_loss,
                                args=(X,Loss,lock,count,X.shape[0],batch_num)) for _ in range(proc_num)]
        [proc.start() for proc in proc_list]
        [proc.join() for proc in proc_list]
        return np.array(Loss[:],dtype=ctypes.c_double)
    
    def parallel_rotation_loss(self,X:np.ndarray,proc_num:int,batch_num:int):
        proc_num = os.cpu_count() if proc_num <0 else proc_num
        parallel = Parallel(n_jobs=proc_num,batch_size=batch_num)
        loss_res = parallel(delayed(self.rotation_loss)(x) for x in X)
        return np.array(loss_res,dtype=ctypes.c_double)
    
    def LSM(self,R:np.ndarray):
        AA = []
        BB = []
        for cedge,pedge in zip(self.camera_edge,self.pcd_edge):
            AA.append(np.hstack((cedge[:3,:3]-np.eye(3),cedge[:3,[3]])))  # (3,4)
            BB.append(R @ pedge[:3,[3]])  # (3,1)
        AA, BB = np.vstack(AA), np.vstack(BB)  # (3N, 4), (3N, 1)
        sol = np.linalg.solve(AA.T @ AA, AA.T @ BB)
        return sol[:3].reshape(-1), sol[-1]  # t, s


class WeightedEdgeLoss:
    def __init__(self,camera_edge,pcd_edge,weight,init_rot:np.eye(3)):
        assert len(camera_edge) == len(pcd_edge) == len(weight), "camera edge (%d) != pcd edge (%d) or weight (%d)"%(len(camera_edge),len(pcd_edge),len(weight))
        self.edge_num = len(camera_edge)
        self.camera_edge = camera_edge
        self.pcd_edge = pcd_edge
        self.weight = weight
        self.init_rot = init_rot
        
    def transform_loss(self,se3_tran_:np.ndarray,scale,reduction='mean'):
        init_tran = np.eye(4)
        init_tran[:3,:3] = self.init_rot
        se3_tran = se3_tran_.dot(init_tran)
        loss_list = np.zeros([self.edge_num,6])
        for i in range(self.edge_num):
            camera_tran = self.camera_edge[i].copy()
            camera_tran[:3,3] *= scale
            error_se3 = (np.linalg.inv(se3_tran @ self.pcd_edge[i]) @ camera_tran @ se3_tran)*self.weight[i]
            loss_list[i,:3] = toVec(error_se3[:3,:3])
            loss_list[i,3:] = error_se3[:3,3]
        if reduction == 'mean':
            return loss_list.mean(axis=0)
        elif reduction == 'none':
            return loss_list
        elif reduction == 'sum':
            return loss_list.sum(axis=0)
        else:
            warnings.warn('Unknown reduction, return mean instead',RuntimeWarning)
            return loss_list.mean(axis=0)
    
    def update_weight(self,mu:float,rot_vec:np.ndarray):
        trloss = self.rotation_loss(rot_vec,reduction='none')**2
        self.weight = (mu/(mu+trloss))**2
    
    def rotation_loss(self,rot_vec:np.ndarray,reduction='mean'):
        loss = np.zeros(self.edge_num)
        RCL = vec_tran(rot_vec).dot(self.init_rot)
        for i,(cedge,pedge) in enumerate(zip(self.camera_edge,self.pcd_edge)):
            RC= cedge[:3,:3]
            RL = pedge[:3,:3]
            Rerr = RC @ RCL - RCL @ RL 
            loss[i] = np.sqrt(np.sum(Rerr**2))
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            warnings.warn('Unknown reduction, return mean instead',RuntimeWarning)
            return loss.mean()
    
    def procfun_rotation_loss(self,X:np.ndarray,Y:mp.Array,
                                lock:mp.Lock,count:mp.Value,max_num:int,batch_num:int):
        while True:
            with lock:
                local_cnt = count.value
                count.value += batch_num
            if local_cnt >= max_num:
                break
            end_index = min(local_cnt+batch_num,max_num)
            for idx in range(local_cnt,end_index):
                x = X[idx]
                loss = self.rotation_loss(x)
                Y[idx] = loss
    
    def proc_rotation_loss(self,X:np.ndarray,proc_num:int,batch_num:int):
        proc_num = os.cpu_count() if proc_num <0 else proc_num
        Loss = mp.Array(ctypes.c_double,X.shape[0])
        count = mp.Value(ctypes.c_int)
        lock = mp.Lock()
        proc_list = [mp.Process(target=self.procfun_rotation_loss,
                                args=(X,Loss,lock,count,X.shape[0],batch_num)) for _ in range(proc_num)]
        [proc.start() for proc in proc_list]
        [proc.join() for proc in proc_list]
        return np.array(Loss[:],dtype=ctypes.c_double)
    
    def LSM(self,R:np.ndarray):
        AA = []
        BB = []
        WW = np.zeros([3*self.edge_num,3*self.edge_num])
        for i,(cedge,pedge) in enumerate(zip(self.camera_edge,self.pcd_edge)):
            AA.append(np.hstack((cedge[:3,:3]-np.eye(3),cedge[:3,[3]])))  # (3,4)
            BB.append(R @ pedge[:3,[3]])  # (3,1)
            WW[3*i:3*(i+1),3*i:3*(i+1)] = np.diag(np.ones(3)*self.weight[i])  # (3,3)
        AA, BB = np.vstack(AA), np.vstack(BB)  # (3n,4)  (3n,1)
        sol = np.linalg.inv(AA.T @ WW @ AA) @ AA.T @ WW @ BB
        return sol[:3].reshape(-1), sol[-1]  # t, s


if __name__ == "__main__":
    camera_json_path = "res/tmp/sfm_data.json"
    pcd_json_path = "res/building/optimized.json"
    camera_pose = read_camera_json(camera_json_path)
    pcd_pose = read_pcd_json(pcd_json_path)
    print(len(camera_pose),len(pcd_pose))
    camera_edge = poseToAdjedge(camera_pose)
    pcd_edge = poseToAdjedge(pcd_pose)
    print(len(camera_edge),len(pcd_edge))
    
    print(camera_edge[0],'\n',camera_edge[1])