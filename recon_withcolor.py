import argparse
import os
import open3d as o3d
import numpy as np
import json
from clique_utils import pose_graph_trim, flatten_clique_list
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import time
import yaml

global_set = yaml.load(open("config.yml",'r'),yaml.SafeLoader)
work_dir = global_set['work_dir']
method = global_set['method']
def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def change_background_to_white(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    return False
def capture_img(vis):
    image = vis.capture_screen_float_buffer()
    # image = vis.capture_depth_float_buffer()
    plt.imsave(os.path.join('view',time.strftime("%H-%M-%S",time.localtime())+".png"),np.array(image))
    return False

class PtPainiter:
    def __init__(self, cmap:str='viridis',step=10) -> None:
        self.step = step
        self.cmap = get_cmap(cmap)
        self.colors = self.cmap(np.linspace(0,1,step,endpoint=True))
    def color(self, pt_list:list):
        npcd = len(pt_list)
        bins = np.linspace(0,npcd,self.step)
        for i in range(npcd):
            c = np.digitize(i,bins,right=False)
            colors = np.repeat(self.colors[c][None,:3],len(pt_list[i].points),0)
            pt_list[i].colors = o3d.utility.Vector3dVector(colors)
        

def str2bool(c:str)->bool:
    if c.lower() == "true":
        return True
    elif c.lower == "false":
        return False
    return False

def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default='{work_dir}/pcd'.format(work_dir=work_dir))
    parser.add_argument("--step",type=int,default=1)
    parser.add_argument("--pose_graph",type=str,default="res/{work_dir}/{method}_union.json".format(work_dir=work_dir,method=method))
    parser.add_argument("--trim_graph",type=bool,default=False,help="Set to True if this pose graph need to be trimmed")
    parser.add_argument("--result",type=str,default='res/{work_dir}/{method}_union.pcd'.format(work_dir=work_dir,method=method))
    parser.add_argument("--voxel_size",type=float,default=0.05)
    parser.add_argument("--trim",type=str2bool,default=True)
    parser.add_argument("--clique_file",type=str,default='res/{work_dir}/clique_{method}.json'.format(work_dir=work_dir,method=method))
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
    painter = PtPainiter()

    if os.path.exists(args.clique_file) and args.trim:
        clique_list = json.load(open(args.clique_file,'r'))
        inlier_list = flatten_clique_list(clique_list)
        idx = np.arange(N)
        args.outlier_idx = [i for i in idx if i not in inlier_list]
        print('Outlier idx:{}'.format(args.outlier_idx))
    else:
        # if args.part_index[1] < 0:
        #     args.part_index[1] = N
        # if args.part_index[0] < 0:
        #     args.part_index[1] = 0
        inlier_list = np.arange(N)
        print("Use all nodes")
    
    # poses = read_pcd_pose("res/building_rot3/pose")
    # for idx in range(N):
    #     pcd = o3d.io.read_point_cloud(os.path.join(args.input_dir,pcd_filenames[idx]))
    #     pcd = pcd.transform(poses[idx])
    #     scene_pcd_arr = np.vstack((scene_pcd_arr,np.array(pcd.points)))
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("B")] = change_background_to_white
    key_to_callback[ord(".")] = capture_img
    pose_graph = o3d.io.read_pose_graph(args.pose_graph)
    if args.trim_graph:
        pose_graph = pose_graph_trim(pose_graph,inlier_list)
    pcd_filenames = [filename for i,filename in enumerate(pcd_filenames) if i in inlier_list]
    pcd_list = []
    for idx in range(len(pcd_filenames)):
        # if idx in args.outlier_idx:
        #     continue
        pcd = o3d.io.read_point_cloud(os.path.join(args.input_dir,pcd_filenames[idx]))
        pcd.transform(pose_graph.nodes[idx].pose)
        pcd_list.append(pcd)
    painter.color(pcd_list)
    o3d.visualization.draw_geometries_with_key_callbacks(pcd_list,key_to_callback)
    # o3d.io.write_point_cloud(args.result,scene_pcd)