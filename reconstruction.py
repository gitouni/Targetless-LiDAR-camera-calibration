import argparse
import os
import open3d as o3d
import numpy as np
import json
from clique_utils import pose_graph_trim, flatten_clique_list
import yaml

global_set = yaml.load(open("config.yml",'r'),yaml.SafeLoader)

        

def str2bool(c:str)->bool:
    if c.lower() == "true":
        return True
    elif c.lower == "false":
        return False
    return False

def input_args():
    res_dir = global_set['res_dir']
    work_dir = global_set['work_dir']
    method = global_set['method']
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default='data/proc_pcd')
    parser.add_argument("--step",type=int,default=1)
    parser.add_argument("--pose_graph",type=str,default="{res_dir}/{work_dir}/{method}_union.json".format(res_dir=res_dir,work_dir=work_dir,method=method))
    parser.add_argument("--trim_graph",type=bool,default=False,help="Set to True if this pose graph need to be trimmed")
    parser.add_argument("--result",type=str,default='{res_dir}/{work_dir}/{method}_union.pcd'.format(res_dir=res_dir,work_dir=work_dir,method=method))
    parser.add_argument("--voxel_size",type=float,default=0.05)
    parser.add_argument("--trim",type=str2bool,default=True)
    parser.add_argument("--clique_file",type=str,default='{res_dir}/{work_dir}/clique_{method}.json'.format(res_dir=res_dir,work_dir=work_dir,method=method))
    parser.add_argument("--outlier_idx",type=int,nargs="+",default=[])  # 16,47,116
    parser.add_argument("--part_index",nargs=2, type=int, default=[-1,-1])
    
    return parser.parse_args()

if __name__ == "__main__":
    args = input_args()
    pcd_filenames = list(sorted(os.listdir(args.input_dir)))
    N = len(pcd_filenames)
    index = np.linspace(0,N-1,N//args.step,endpoint=True).astype(np.int32)
    pcd_filenames = [pcd_filenames[idx] for idx in index]
    N = len(pcd_filenames)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd_arr = np.empty((0,3),dtype=np.float32)

    if os.path.exists(args.clique_file) and args.trim:
        clique_list = json.load(open(args.clique_file,'r'))
        inlier_list = flatten_clique_list(clique_list)
        idx = np.arange(N)
        args.outlier_idx = [i for i in idx if i not in inlier_list]
        print('Outlier idx:{}'.format(args.outlier_idx))
    else:
        if args.part_index[1] < 0:
            args.part_index[1] = N
        if args.part_index[0] < 0:
            args.part_index[1] = 0
        inlier_list = np.arange(*args.part_index)
        print("Use all nodes")
    
    # poses = read_pcd_pose("res/building_rot3/pose")
    # for idx in range(N):
    #     pcd = o3d.io.read_point_cloud(os.path.join(args.input_dir,pcd_filenames[idx]))
    #     pcd = pcd.transform(poses[idx])
    #     scene_pcd_arr = np.vstack((scene_pcd_arr,np.array(pcd.points)))
    pose_graph = o3d.io.read_pose_graph(args.pose_graph)
    if args.trim_graph:
        pose_graph = pose_graph_trim(pose_graph,inlier_list)
    pcd_filenames = [filename for i,filename in enumerate(pcd_filenames) if i in inlier_list]
    for idx in range(len(pcd_filenames)):
        # if idx in args.outlier_idx:
        #     continue
        pcd = o3d.io.read_point_cloud(os.path.join(args.input_dir,pcd_filenames[idx]))
        pcd.transform(pose_graph.nodes[idx].pose)
        scene_pcd_arr = np.vstack((scene_pcd_arr,np.array(pcd.points,dtype=np.float32)))
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_arr)
    scene_pcd = scene_pcd.voxel_down_sample(args.voxel_size)
    # scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*6,max_nn=30))
    # scene_pcd.normalize_normals()
    # scene_pcd.normals = o3d.utility.Vector3dVector(np.array(scene_pcd.normals)/5)
    
    o3d.visualization.draw_geometries([scene_pcd],point_show_normal=False)
    # o3d.io.write_point_cloud(args.result,scene_pcd)