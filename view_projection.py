import cv2
import os
import open3d as o3d
import numpy as np
from PIL import Image
# import matplotlib
# matplotlib.use('agg') // uncomment these lines if you do not have X11 service
from matplotlib import pyplot as plt
import argparse
import yaml

global_set = yaml.load('config.yml',yaml.SafeLoader)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",type=int,default=77)
    parser.add_argument("--extrinsic_file",type=str,default="ranreg_6dof.txt")
    parser.add_argument("--img_dir",type=str,default="/data/img")
    parser.add_argument("--pcd_dir",type=str,default="/data/proc_pcd")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    index = args.index
    pcd_dir = args.pcd_dir
    img_dir = args.img_dir
    extrinsic_pose_file = "ransac_6dof2_init.txt"  # TCL
    rawpose_file = "TCL2.txt"
    distort_params = np.array([-0.11067364734378947,0.13514364456382348,0,0,0.045611917550973459])
    ins = np.loadtxt(os.path.join(os.path.dirname(__file__),"K.txt")).T
    pcd_files = list(sorted(os.listdir(pcd_dir)))
    img_files = list(sorted(os.listdir(img_dir)))
    pcd_file = os.path.join(pcd_dir,pcd_files[index])
    img_file = os.path.join(img_dir,img_files[index])
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd_arr = np.array(pcd.points).T
    img = np.array(Image.open(img_file))
    img = cv2.undistort(img,ins,distort_params)
    TLC = np.loadtxt(os.path.join(os.path.dirname(__file__),args.extrinsic_file),delimiter=' ')
    TCL = np.linalg.inv(TLC)
    proj_pcd:np.ndarray = TCL[:3,:3] @ pcd_arr + TCL[:3,[3]]
    proj_pcd = ins.dot(proj_pcd)
    H,W = img.shape[:2]
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = u/w
    v = v/w
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = np.linalg.norm(pcd_arr[:,rev],axis=0)
    plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    plt.imshow(img)
    plt.scatter([u],[v],c=[r],cmap='rainbow_r',alpha=0.8,s=3)
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__),'demo_tcl_{:04d}.png'.format(index)),bbox_inches='tight')
