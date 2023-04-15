import open3d as o3d
import os
import argparse
import numpy as np
from tqdm import tqdm
import json
from clique_utils import pose_graph_trim
from copy import deepcopy
import shutil
import yaml

global_set = yaml.load(open("config.yml",'r'),yaml.SafeLoader)
work_dir = global_set['work_dir']
method = global_set['method']

def print_info(info,**argv):
    print("\033[1;34m{:s}\033[0m".format(info),**argv)

def print_highlight(info,**argv):
    print('\033[1;32m{:s}\033[0m'.format(info),**argv)
    
def print_warning(info,**argv):
    print('\033[1;33m{:s}\033[0m'.format(info),**argv)
    
def print_error(info,**argv):
    print('\033[1;31m{:s}\033[0m'.format(info),**argv)
    
def input_args():
    parser = argparse.ArgumentParser()
    input_parser = parser.add_argument_group()
    input_parser.add_argument("--work_dir",type=str,default=work_dir)
    input_parser.add_argument("--res_dir",type=str,default="res")
    input_parser.add_argument("--input_dir",type=str,default='pcd')
    input_parser.add_argument("--clique_file",type=str,default="clique_{}.json".format(method))
    input_parser.add_argument("--init_pose_graph",type=str,default='{}_raw.json'.format(method))
    input_parser.add_argument("--step",type=int,default=1)
    
    output_parser = parser.add_argument_group()
    output_parser.add_argument("--pose_graph_dir",type=str,default="{}_cliques".format(method))
    output_parser.add_argument("--clique_desc",type=str,default="clique_desc_{}.json".format(method))
    output_parser.add_argument("--pose_graph_fmt",type=str,default='clique_{:02d}.json')
    output_parser.add_argument("--voxel_size",type=float,default=0.05)
    output_parser.add_argument("--radius",type=float,default=0.6)  # 1.0 for floam
    
    args = parser.parse_args()
    work_dir = os.path.join(args.res_dir,args.work_dir)
    os.makedirs(work_dir,exist_ok=True)
    args.input_dir = os.path.join(args.work_dir,args.input_dir)
    args.init_pose_graph, args.clique_file, args.pose_graph_dir, args.clique_desc \
        = map(lambda file: os.path.join(work_dir,file),[args.init_pose_graph, args.clique_file, args.pose_graph_dir,args.clique_desc])
    return args



def full_graph_refine(pcds, raw_pose_graph, radius:float):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    print('creating Pose Graph...')
    tqdm_consle = tqdm(total=n_pcds)
    with tqdm_consle:
        for source_id in range(n_pcds):
            tqdm_consle.set_postfix_str("{}|{}".format(source_id+1,n_pcds))
            for target_id in range(source_id + 1, n_pcds):
                src_pcd = deepcopy(pcds[source_id])
                tgt_pcd = deepcopy(pcds[target_id])
                tqdm_consle.set_postfix_str("Sub:{}|{}: register {} points to {} points".format(
                    target_id-source_id,n_pcds-source_id,len(src_pcd.points),len(tgt_pcd.points)))
                init_tran = np.linalg.inv(raw_pose_graph.nodes[target_id].pose) @ raw_pose_graph.nodes[source_id].pose
                res = o3d.pipelines.registration.registration_icp(
                    src_pcd,tgt_pcd,radius,init_tran,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                transformation_icp = res.transformation
                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    src_pcd,
                    tgt_pcd,
                    radius,
                    transformation_icp
                )
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
    if os.path.exists(args.pose_graph_dir):
        print_warning("Existing Pose Directory {} will be removed".format(args.pose_graph_dir))
        shutil.rmtree(args.pose_graph_dir)
    os.makedirs(args.pose_graph_dir)
    pcd_filenames = list(sorted(os.listdir(args.input_dir)))
    N = len(pcd_filenames)
    index = np.linspace(0,N-1,N//args.step,endpoint=True).astype(np.int32)
    pcd_filenames = [pcd_filenames[idx] for idx in index]
    N = len(pcd_filenames)
    init_pose_graph = o3d.io.read_pose_graph(args.init_pose_graph)
    clique_list:list = json.load(open(args.clique_file))
    clique_list.sort()
    clique_dict = dict(clique_list=clique_list,\
        pose_graph=[args.pose_graph_fmt.format(idx) for idx in range(len(clique_list))],\
        pose_graph_dir=args.pose_graph_dir)
    json.dump(clique_dict,open(args.clique_desc,'w'),indent=4)
    print("Clique List:\n{}".format(clique_list))
    for i,clique in enumerate(clique_list):
        pose_graph = pose_graph_trim(init_pose_graph, clique)
        pcd_list = []
        for j in tqdm(clique,desc='Loading pcds'):
            filename = os.path.join(args.input_dir,pcd_filenames[j])
            pcd = o3d.io.read_point_cloud(filename)
            pcd_list.append(pcd)
        pose_graph = full_graph_refine(pcd_list, pose_graph, args.radius)
        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.radius,
            edge_prune_threshold=5*args.radius,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        o3d.io.write_pose_graph(os.path.join(args.pose_graph_dir,args.pose_graph_fmt.format(i)),pose_graph)