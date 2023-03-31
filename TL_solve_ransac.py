from copy import deepcopy
import json
from RANSAC.base import RotEstimator,TslEstimator,RotRANSAC, TslRANSAC
import numpy as np
# from utils.transform import RandomTransformSE3
from problem import read_camera_json,read_pcd_json,fullEdge_Idx
import argparse
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
work_dir = "building_imu"
method = "ranreg"
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_json",type=str,default="res/{work_dir}/sfm_data.json".format(work_dir=work_dir))
    parser.add_argument("--pcd_json",type=str,default="res/{work_dir}/{method}_raw.json".format(work_dir=work_dir,method=method))
    ransac_parser = parser.add_argument_group()
    ransac_parser.add_argument("--scale_threshold",type=float,default=0.15)  # floam for 0.25
    ransac_parser.add_argument("--max_iter",type=int,default=20000)
    ransac_parser.add_argument("--min_sample",type=int,default=3)
    ransac_parser.add_argument("--stop_prob",type=float,default=0.999)
    ransac_parser.add_argument("--se3_threshold",type=float,default=2.0)
    ransac_parser.add_argument("--clique_save",type=str,default="res/{work_dir}/clique_{method}.json".format(work_dir=work_dir,method=method))
    ransac_parser.add_argument("--sol_save",type=str,default="res/{work_dir}/TL_{method}_rsol".format(work_dir=work_dir,method=method))
    ransac_parser.add_argument("--seed",type=int,default=0) # 0
    return parser.parse_args()


class RANSACClique:
    def __init__(self,camera_json:str,pcd_json:str,
                 max_iter:int,
                 min_sample:int,
                 stop_prob:float,
                 scale_threshold:float,
                 seed:0):
        self.raw_camera_pose = np.array(read_camera_json(camera_json))
        self.raw_pcd_pose = np.array(read_pcd_json(pcd_json))
        assert len(self.raw_camera_pose) == len(self.raw_pcd_pose)
        self.Npose = len(self.raw_camera_pose)
        self.raw_pose_idx = np.arange(self.Npose)
        self.cur_pose_idx = deepcopy(self.raw_pose_idx)
        self.cur_camera_pose = deepcopy(self.raw_camera_pose)
        self.cur_pcd_pose = deepcopy(self.raw_pcd_pose)
        self.pose_inlier = np.arange(self.Npose)
        self.cur_pose_inlier = deepcopy(self.pose_inlier)
        self.clique_list = []
        self.max_iter = max_iter
        self.min_sample = min_sample
        self.stop_prob = stop_prob
        self.scale_threshold = scale_threshold
        self.seed = seed
        self.inlier_scale = 1.0
        self.inlier_rot = np.eye(3)
        self.inlier_tsl = np.zeros(3)
        self.dbscan = DBSCAN(metric='precomputed')
        
    def ransac_solve(self):
        camera_edge, _ = fullEdge_Idx(self.cur_camera_pose)
        pcd_edge, edge_idx = fullEdge_Idx(self.cur_pcd_pose)
        edge_idx = np.array(edge_idx) # (N*(N-1)/2,2)
        idx = np.arange(edge_idx.shape[0])
        Npose = len(self.cur_camera_pose)
        edge_matrix = np.zeros([Npose,Npose])
        camera_edge, pcd_edge = map(lambda edge: np.array(edge),[camera_edge,pcd_edge])
        camera_rotedge, pcd_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge,pcd_edge])
        ransac_rot_estimator = RotRANSAC(RotEstimator(),
                                min_samples=self.min_sample,
                                max_trials=self.max_iter,
                                stop_prob=self.stop_prob,
                                random_state=self.seed)
        alpha,beta = map(ransac_rot_estimator.toVecList,[camera_rotedge,pcd_rotedge])
        best_rot, rot_inlier_mask = ransac_rot_estimator.fit(beta,alpha)
        idx = idx[rot_inlier_mask]
        ransac_tsl_estimator = TslRANSAC(TslEstimator(best_rot),
                                     min_samples=self.min_sample,
                                     max_trials=self.max_iter,
                                     stop_prob=self.stop_prob,
                                     random_state=self.seed)
        camera_flatten = ransac_tsl_estimator.flatten(camera_edge[rot_inlier_mask])
        pcd_flatten = ransac_tsl_estimator.flatten(pcd_edge[rot_inlier_mask])
        best_tsl, best_scale, tsl_inlier_mask = ransac_tsl_estimator.fit(camera_flatten,pcd_flatten)
        pred_pcd_flatten = ransac_tsl_estimator.ransac_estimator.predict(camera_flatten)
        inv_pred_pcd_se3 = np.array([np.linalg.inv(pred.reshape(4,4)) for pred in pred_pcd_flatten])
        err = np.mean(np.abs(inv_pred_pcd_se3 @ pcd_flatten.reshape(-1,4,4)-np.eye(4)[None,...]))
        idx = idx[tsl_inlier_mask]
        edge_matrix[edge_idx[idx,0],edge_idx[idx,1]] = 1
        edge_matrix = edge_matrix + edge_matrix.T + np.eye(Npose)
        self.dbscan.fit(1-edge_matrix)
        pose_inlier_mask = self.dbscan.labels_!=-1
        return best_rot, best_tsl, best_scale, pose_inlier_mask, err
    
    def clique_solve(self,threshold=0.1,max_iter=10,min_clique_num=3):
        err = 0
        for iteration in range(1,1+max_iter):
            best_rot, best_tsl, best_scale, pose_inlier_mask, err = self.ransac_solve()
            inlier_pose = deepcopy(self.cur_pose_idx[pose_inlier_mask])
            self.cur_pose_idx = self.cur_pose_idx[~pose_inlier_mask]
            self.cur_camera_pose = self.cur_camera_pose[~pose_inlier_mask]
            self.cur_pcd_pose = self.cur_pcd_pose[~pose_inlier_mask]
            clique_num = pose_inlier_mask.sum()
            left_num = np.logical_not(pose_inlier_mask).sum()
            print("Iter {}:\nRotation:{}\nTranslation:{}\nScale:{}\nTotal:{} Clique:{} Left:{}".format(
                iteration,best_rot,best_tsl,best_scale,self.Npose,clique_num,left_num
            ))
            if clique_num < min_clique_num:
                print("Clique num {} < {}, break down.".format(clique_num,min_clique_num))
                break
            if iteration == 1:
                self.inlier_scale = best_scale
                self.inlier_rot = best_rot
                self.inlier_tsl = best_tsl
            else:
                if not (1 - self.scale_threshold < (self.inlier_scale/best_scale) < 1 + self.scale_threshold):
                    print('Scale changed too much cur | inlier: {} | {}, break down.'.format(self.inlier_scale,best_scale))
                    break
            if err >= threshold:
                print("Error {} >  {}, break down".format(err,threshold))
                break
            print("Inlier se3 error:{}".format(err))
            self.clique_list.append(inlier_pose.tolist())
            if len(self.cur_pcd_pose) == 0:
                print("No pose left!")
                break
        
        
if __name__ == "__main__":
    args = options()
    clique_solver = RANSACClique(args.camera_json,args.pcd_json,
                                 args.max_iter,args.min_sample,args.stop_prob,args.scale_threshold,
                                 args.seed)
    clique_solver.clique_solve(threshold=args.se3_threshold,min_clique_num=3)
    print(clique_solver.clique_list)
    json.dump(clique_solver.clique_list,open(args.clique_save,'w'))
    np.savez(args.sol_save,\
        rotation=clique_solver.inlier_rot,translation=clique_solver.inlier_tsl,scale=clique_solver.inlier_scale)