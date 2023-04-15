from functools import partial
import open3d as o3d
import os
import argparse
from ransac import register_with_feature as ransac_register
import numpy as np 
from tqdm import tqdm
from view.view_normal import Surface_TNormal
import yaml

global_set = yaml.load(open("config.yml",'r'),yaml.SafeLoader)

def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir",type=str,default=global_set['work_dir'])
    parser.add_argument("--res_dir",type=str,default=global_set['res_dir'])
    parser.add_argument("--input_dir",type=str,default='/data/proc_pcd')
    parser.add_argument("--step",type=int,default=1)
    parser.add_argument("--pose_graph",type=str,default='{}_raw.json'.format(global_set['method']))
    parser.add_argument("--voxel_size",type=float,default=0.15)
    parser.add_argument("--radius",type=float,default=0.3)
    parser.add_argument("--knn_num",type=int,default=30)
    parser.add_argument("--ne_method",type=str,default="defined",choices=['o3d','defined'])
    parser.add_argument("--feat_method",type=str,default='FPFH',choices=['FPFH'])
    parser.add_argument("--method",type=str,default="RANSAC",choices=['ICP','RANSAC','FGR'])
    parser.add_argument("--seed",type=int,default=None)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.res_dir,args.basedir),exist_ok=True)
    args.input_dir = os.path.join(args.basedir,args.input_dir)
    args.pose_graph = os.path.join(args.res_dir,args.basedir,args.pose_graph)
    return args

def compute_fpfh_with_method(pcd:o3d.geometry.PointCloud,method):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,method)
    return pcd_fpfh


class Register:
    def __init__(self,fpfh_radius:float,distance_thre:float,
                 max_iter:int=64,max_tuple:int=1000,icp_tuned:bool=False,discentral=False,compute_feat_func=None,seed=None):
        self.fpfh_radius = fpfh_radius
        self.seed= seed
        self.distance_thre = distance_thre
        self.max_iter = max_iter
        self.max_tuple = max_tuple
        self.icp_tuned = icp_tuned
        self.g0p0 = np.eye(4)
        self.g0p1 = np.eye(4)
        self.trans = np.eye(4)
        self.discentral = discentral
        self.method = {"ICP":self.icp_register,"FGR":self.fgr_register,
                       "RANSAC":self.ransac_register}
        self.compute_feat_func = compute_feat_func
    
    def zero_mean(self, p0:np.ndarray, p1:np.ndarray):
        """zero_mean for pcd pairs

        Args:
            p0 (np.ndarray): source pcd
            p1 (np.ndarray): template pcd
        """
        p0_mean = np.mean(p0,axis=0,keepdims=True)  # (1,3)
        p1_mean = np.mean(p1,axis=0,keepdims=True)  # (1,3)
        p0 -= p0_mean
        p1 -= p1_mean
        self.g0p0[:3,3] = -p0_mean.reshape(-1)
        self.g0p1[:3,3] = -p1_mean.reshape(-1)
        return p0, p1
    
    def compute_fpfh(self,pcd:o3d.geometry.PointCloud):
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
            # o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius,max_nn=100)
             o3d.geometry.KDTreeSearchParamKNN(30)
            )
        return pcd_fpfh
    
    def icp_register(self,src_pcd,dst_pcd,src_feat,dst_feat) -> np.ndarray:
        if self.discentral:
            src_pcd_arr, dst_pcd_arr = self.zero_mean(src_pcd_arr, dst_pcd_arr)
            src_pcd.points = o3d.utility.Vector3dVector(src_pcd_arr)
            dst_pcd.points = o3d.utility.Vector3dVector(dst_pcd_arr)
        res = o3d.pipelines.registration.registration_icp(
            src_pcd,dst_pcd,self.distance_thre,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        self.trans = res.transformation
        return self.trans
        
    def ransac_register(self,src_pcd,dst_pcd,src_feat,dst_feat) -> np.ndarray:
        self.trans = ransac_register(src_pcd,dst_pcd,src_feat,dst_feat,icp_tuned=self.icp_tuned,seed=self.seed)
        return self.trans
    
    def fgr_register(self,src_pcd,dst_pcd,src_feat,dst_feat) -> np.ndarray:
        src_pcd_arr, dst_pcd_arr = map(lambda pcd: np.array(pcd.points),[src_pcd,dst_pcd])
        if self.discentral:
            src_pcd_arr, dst_pcd_arr = self.zero_mean(src_pcd_arr, dst_pcd_arr)
            src_pcd.points = o3d.utility.Vector3dVector(src_pcd_arr)
            dst_pcd.points = o3d.utility.Vector3dVector(dst_pcd_arr)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_pcd, dst_pcd, src_feat,dst_feat)
        est_g = result.transformation
        if self.icp_tuned:
            result = o3d.pipelines.registration.registration_icp(
            src_pcd, dst_pcd, self.distance_thre, est_g,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            est_g = result.transformation
        self.trans = np.linalg.inv(self.g0p1) @ est_g @ self.g0p0 
        return self.trans
    
    def register_with_info(self, src_pcd:o3d.geometry.PointCloud,dst_pcd:o3d.geometry.PointCloud,
                           src_feat: o3d.pipelines.registration.Feature, dst_feat:o3d.pipelines.registration.Feature,
                           radius:float, method:str):
        method = method.upper()
        assert method in self.method.keys()
        self.method[method](src_pcd,dst_pcd,src_feat,dst_feat)

        info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd,
            dst_pcd,
            radius,
            self.trans
        )
        return self.trans, info_matrix, 
    
def print_info(info,**argv):
    print("\033[1;34m{:s}\033[0m".format(info),**argv)

def print_highlight(info,**argv):
    print('\033[1;32m{:s}\033[0m'.format(info),**argv)
    
def print_warning(info,**argv):
    print('\033[1;33m{:s}\033[0m'.format(info),**argv)
    
def print_error(info,**argv):
    print('\033[1;31m{:s}\033[0m'.format(info),**argv)

def preprocess_pcd(pcd:o3d.geometry.PointCloud,o3d_method):
    pcd.estimate_normals(
       o3d_method)  # used for parse point cloud
    return pcd

def defined_preprocess_pcd(pcd:o3d.geometry.PointCloud,ne:Surface_TNormal):
    ne.estimate_normals(pcd)
    return pcd


def full_registration(pcds, pcd_feats, model:Register, voxel:float, method:str):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    print_highlight('creating Pose Graph...')
    tqdm_consle = tqdm(total=n_pcds)
    with tqdm_consle:
        for source_id in range(n_pcds-1):
            for target_id in range(source_id + 1, n_pcds):
                tqdm_consle.set_postfix_str("Sub:{}|{}: register {} points to {} points".format(
                    target_id-source_id,n_pcds-source_id,len(pcds[source_id].points),len(pcds[target_id].points)))
                transformation_icp, information_icp = model.register_with_info(
                    pcds[source_id], pcds[target_id],pcd_feats[source_id],pcd_feats[target_id],2*voxel,method)
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
            tqdm_consle.update(1)
    return pose_graph



if __name__ == "__main__":
    args = input_args()
    pcd_filenames = list(sorted(os.listdir(args.input_dir)))
    N = len(pcd_filenames)
    index = np.linspace(0,N-1,N//args.step,endpoint=True).astype(np.int32)
    pcd_filenames = [pcd_filenames[idx] for idx in index]
    N = len(pcd_filenames)
    raw_pcd_list = []
    pcd_feat_list = []
    o3d_knn_method = o3d.geometry.KDTreeSearchParamKNN(30)
    # o3d_knn_method = o3d.geometry.KDTreeSearchParamHybrid(args.radius,args.knn_num)
    if args.ne_method == 'o3d':
        process_func = partial(preprocess_pcd,o3d_method=o3d_knn_method)
    else:
        ne = Surface_TNormal(radius=args.radius,n_neighbors=args.knn_num,workers=-1,method='hybrid')
        process_func = partial(defined_preprocess_pcd,ne=ne)
    fpfh_knn_method = o3d.geometry.KDTreeSearchParamKNN(args.knn_num)
    compute_feat = partial(compute_fpfh_with_method,method=fpfh_knn_method)
    register = Register(fpfh_radius=args.voxel_size*5,distance_thre=args.voxel_size*2,compute_feat_func=compute_feat,icp_tuned=False,seed=args.seed)
    for filename in tqdm(pcd_filenames,desc='Esitmating Normals'):
        filename = os.path.join(args.input_dir,filename)
        pcd = o3d.io.read_point_cloud(filename)
        if args.feat_method == "FPFH":
            raw_pcd_list.append(process_func(pcd))
        else:
            raw_pcd_list.append(pcd)
    for pcd in tqdm(raw_pcd_list,desc='Computing {} Features'.format(args.feat_method)):
        pcd_feat_list.append(compute_feat(pcd))
    print_highlight("{} pcds loaded, start pairwise registration.".format(N))
    # with o3d.utility.VerbosityContextManager(
    #     o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(raw_pcd_list,pcd_feat_list,register,args.voxel_size,args.method)
    # pose_graph = full_registration_multiporc(raw_pcd_list,register,args.voxel_size,16,4)
    print_highlight("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=2*args.radius,
        edge_prune_threshold=5*args.radius,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    o3d.io.write_pose_graph(args.pose_graph,pose_graph)

