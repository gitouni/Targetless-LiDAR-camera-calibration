import open3d as o3d
from copy import deepcopy
import numpy as np


def pose_graph_trim(graph:o3d.pipelines.registration.PoseGraph, revPoseIdx:list):
    rev_nodes = [node for i,node in enumerate(graph.nodes) if i in revPoseIdx]
    rev_edges = list()
    for edge in graph.edges:
        if (edge.source_node_id in revPoseIdx) or (edge.target_node_id in revPoseIdx):
            rev_edges.append(deepcopy(edge))
    new_graph = o3d.pipelines.registration.PoseGraph()
    for node in rev_nodes:
        new_graph.nodes.append(node)
    for edge in rev_edges:
        new_graph.edges.append(edge)
    return new_graph


def pose_graph_merge(graph_list:list, clique_list:list):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    node_list = []
    idx_list = []
    edge_list = []
    for graph,clique in zip(graph_list,clique_list):
        node_list += graph.nodes
        idx_list += clique
    sort_list = list(sorted(range(len(idx_list)),key=lambda i:idx_list[i]))
    for graph in graph_list:
        for edge in graph.edges:
            edge_ = deepcopy(edge)
            edge_.source_node_id = sort_list.index(edge_.source_node_id)
            edge_.target_node_id = sort_list.index(edge_.target_node_id)
            edge_list.append(edge_)
    for i in sort_list:
        pose_graph.nodes.append(node_list[i])
    for edge in edge_list:
        pose_graph.edges.append(edge)
    return pose_graph

def pose_graph_ref(pose_graph,reference:int):
    ref_pose = np.linalg.inv(pose_graph.nodes[reference].pose)
    for i in range(len(pose_graph.nodes)):
        pose_graph.nodes[i].pose = ref_pose @ pose_graph.nodes[i].pose
    return pose_graph

def flatten_clique_list(clique_list:list):
    inlier_list = []
    for clique in clique_list:
        inlier_list += clique
    inlier_list.sort()
    return inlier_list

def detect_clique(clique_idx:list,clique_list:list,min_num:int,max_num:int):
    clique_idx.sort()
    tmp_idx = clique_idx[0]
    tmp_clique = []
    for idx in clique_idx:
        if idx <= tmp_idx + 1:
            tmp_clique.append(idx)
        else:
            if min_num <= len(tmp_clique) <= max_num + min_num: 
                clique_list.append(deepcopy(tmp_clique))
            elif len(tmp_clique) > max_num + min_num:
                while len(tmp_clique) > max_num + min_num:
                    clique_list.append(deepcopy(tmp_clique[:max_num]))
                    tmp_clique = tmp_clique[max_num:]
                clique_list.append(tmp_clique)
            tmp_clique = [idx]
        tmp_idx = idx
    if min_num <= len(tmp_clique) <= max_num + min_num: 
        clique_list.append(deepcopy(tmp_clique))
    elif len(tmp_clique) > max_num + min_num:
        while len(tmp_clique) > max_num + min_num:
            clique_list.append(deepcopy(tmp_clique[:max_num]))
            tmp_clique = tmp_clique[max_num:]
        clique_list.append(tmp_clique)
    return clique_list
    
def detect_continous_clique(idx_mlist:list,min_num=3,max_num=10):
    clique_list = []
    for idx_list in idx_mlist:
        clique_list = detect_clique(idx_list,clique_list,min_num,max_num)
    clique_list.sort()
    return clique_list


def detect_clique2(clique_idx:list,clique_list:list,min_num:int,max_num:int):
    clique_idx.sort()
    tmp_clique = deepcopy(clique_idx)
    if min_num <= len(tmp_clique) <= max_num + min_num: 
        clique_list.append(tmp_clique)
    elif len(tmp_clique) > max_num + min_num:
        while len(tmp_clique) > max_num + min_num:
            clique_list.append(deepcopy(tmp_clique[:max_num]))
            tmp_clique = tmp_clique[max_num:]
        clique_list.append(tmp_clique)
    return clique_list
    
def detect_list_clique(idx_mlist:list,min_num=3,max_num=10):
    clique_list = []
    for idx_list in idx_mlist:
        clique_list = detect_clique2(idx_list,clique_list,min_num,max_num)
    clique_list.sort()
    return clique_list

def clique_pcd(pcd_list:list, clique_list:list, voxel_size=0.05):
    fuse_pcd_list = []
    for clique in clique_list:
        fuse_pcd = o3d.geometry.PointCloud()
        scene_pcd_list = []
        for idx in clique:
            scene_pcd_list.append(np.array(pcd_list[idx].points))
        fuse_pcd.points = o3d.utility.Vector3dVector(np.vstack(scene_pcd_list))
        fuse_pcd = fuse_pcd.voxel_down_sample(voxel_size)
        fuse_pcd_list.append(fuse_pcd)
    return fuse_pcd_list

def merge_pcd(pcd_list:list,voxel_size=0.05):
    fuse_pcd_arr = []
    fuse_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        fuse_pcd_arr.append(np.array(pcd.points))
    fuse_pcd.points = o3d.utility.Vector3dVector(np.vstack(fuse_pcd_arr))
    fuse_pcd = fuse_pcd.voxel_down_sample(voxel_size)
    return fuse_pcd
        
if __name__ == "__main__":
    clique_id = [[0,1,2,3,4],list(range(12,30)) + list(range(31,69)),list(range(76,90))]
    clique_list = detect_continous_clique(clique_id)
    print("clique_id:\n{}".format(clique_id))
    print("clique_list:\n{}".format(clique_list))