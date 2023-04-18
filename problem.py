from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import os
import json
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
    alpha -= alpha.mean(axis=0,keepdims=True)
    beta -= beta.mean(axis=0,keepdims=True)
    H = np.dot(beta.T,alpha)  # (3,3)
    U, S, Vt = np.linalg.svd(H)  
    R = np.dot(Vt.T, U.T)     
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    return R, S, alpha, beta


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
    
    def LSM(self,R:np.ndarray):
        AA = []
        BB = []
        for cedge,pedge in zip(self.camera_edge,self.pcd_edge):
            AA.append(np.hstack((cedge[:3,:3]-np.eye(3),cedge[:3,[3]])))  # (3,4)
            BB.append(R @ pedge[:3,[3]])  # (3,1)
        AA, BB = np.vstack(AA), np.vstack(BB)  # (3N, 4), (3N, 1)
        sol = np.linalg.solve(AA.T @ AA, AA.T @ BB)
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