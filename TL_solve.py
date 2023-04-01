import numpy as np
import os
import json
from problem import EdgeLoss, toVec
from problem import read_camera_json,read_pcd_json, poseToAdjedge, poseToFulledge, TL_solve
from scipy.spatial.transform import Rotation
import argparse
from clique_utils import flatten_clique_list

work_dir = "building_imu"
method = "ranreg"
def str2bool(c:str)->bool:
    if c.lower() == "true":
        return True
    elif c.lower == "false":
        return False
    return False

def skew(x:np.ndarray):
    return np.array([[0,-x[2],x[1]],
                     [x[2],0,-x[0]],
                     [-x[1],x[0],0]])


EdgeMethod = dict(adjacent=poseToAdjedge,full=poseToFulledge)
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_TCL_file",type=str,default="res/{}/TCL_review2.txt".format(work_dir))
    parser.add_argument("--camera_json",type=str,default="res/{}/sfm_data.json".format(work_dir))
    parser.add_argument("--pcd_json",type=str,default="res/{}/{}_union.json".format(work_dir,method))
    parser.add_argument("--camera_graph_prune",type=bool,default=True)
    parser.add_argument("--pcd_graph_prune",type=bool,default=False)
    parser.add_argument("--edge_format",type=str,default='full',choices=['full','adjacent'])
    parser.add_argument("--use_clique",type=str2bool,default=True)
    parser.add_argument("--clique_file",type=str,default="res/{}/clique_{}.json".format(work_dir,method))
    parser.add_argument("--save_sol",type=str,default="res/{}/TL_{}_sol.npz".format(work_dir,method))
    parser.add_argument("--max_rot",type=float,default=np.pi)
    parser.add_argument("--proc_num",type=int,default=-1)
    parser.add_argument("--proc_batch",type=int,default=100)
    parser.add_argument("--disp",action='store_true',default=False)
    parser.add_argument("--pso_pop",type=int,default=3000)
    parser.add_argument("--pso_iter",type=int,default=25)
    parser.add_argument("--pso_vratio",type=float,default=0.2)
    parser.add_argument("--pso_eps",type=float,default=1e-8)
    parser.add_argument("--noise_bnd",type=float,default=0.1)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    np.random.seed(0)
    camera_pose = read_camera_json(args.camera_json,toF0=True)
    pcd_pose = read_pcd_json(args.pcd_json)
    camera_pose, pcd_pose = map(lambda pose:np.array(pose),[camera_pose,pcd_pose])
    if os.path.exists(args.clique_file) and args.use_clique:
        print('Existing clique file for trming graph:{}'.format(args.clique_file))
        clique_list = json.load(open(args.clique_file,'r'))
        inlier_list = flatten_clique_list(clique_list)
        inlier_list = np.array(inlier_list)
        if args.camera_graph_prune:
            camera_pose = camera_pose[inlier_list]
        inv_pose0 = np.linalg.inv(camera_pose[0])
        camera_pose = [inv_pose0 @ pose for pose in camera_pose]
        
        if args.pcd_graph_prune:
            pcd_pose = pcd_pose[inlier_list]
            inv_pose0 = np.linalg.inv(pcd_pose[0])
            pcd_pose = [inv_pose0 @ pose for pose in pcd_pose]
    else:
        print("use all poses")
    edge_method = EdgeMethod[args.edge_format]
    camera_edge = edge_method(camera_pose)
    pcd_edge = edge_method(pcd_pose)
    camera_rotedge, pcd_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge,pcd_edge])
    TL_rotation, S, alpha, beta = TL_solve(camera_rotedge,pcd_rotedge)
    Tr_exp = np.loadtxt(args.gt_TCL_file)
    exp_rotation = Tr_exp[:3,:3]
    Loss = EdgeLoss(camera_edge,pcd_edge,np.eye(3))
    exp_rotation_loss = Loss.rotation_loss(toVec(exp_rotation))
    TL_rotation_loss = Loss.rotation_loss(toVec(TL_rotation))
    print('Expected Rotation Loss:{}'.format(exp_rotation_loss))
    print('TL Rotation Loss:{}'.format(TL_rotation_loss))
    exp_tran, exp_scale = Loss.LSM(exp_rotation)
    TL_tran, TL_scale = Loss.LSM(TL_rotation)
    print('Expected Translation:{}, Scale:{}'.format(exp_tran,exp_scale))
    print('TL Translation:{}, Scale:{}'.format(TL_tran,TL_scale))
    print('TL singular values:{}'.format(S))
    print("TL rotation:{}".format(TL_rotation))
    print("Exp rotation:{}".format(exp_rotation))
    Rp = TL_rotation @ beta.T  # (3,N)
    Rp_skew = np.array([skew(Rp[:,i]) for i in range(Rp.shape[1])])  # N of 3x3
    Jacobian = np.zeros(3)
    for i in range(3):
        Jacobian[i] = np.sum(Rp * Rp_skew[:,:,i].T)
    print("Jacobian:{}".format(Jacobian))
    ErrorSO3 = np.linalg.inv(TL_rotation) @ exp_rotation
    tlerror = TL_tran-exp_tran
    rterror = Rotation.from_matrix(ErrorSO3).as_euler("XYZ",degrees=True)
    print("Error Translation (m):{}, RMSE:{}".format(tlerror,np.sqrt((tlerror**2).sum()/3)))
    print("Error Rotation (degree):{}, RMSE:{}".format(rterror, np.sqrt((rterror**2).sum()/3)))
    np.savez(args.save_sol,rotation=TL_rotation,translation=TL_tran,scale=TL_scale)
