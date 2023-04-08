import cv2
import os
import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


sim_pose_file = "TCL2.txt"  # TCL
# rawpose_file = "TCL2.txt"
distort_params = np.array([-0.11067364734378947,0.13514364456382348,0,0,0.045611917550973459])
ins = np.loadtxt(os.path.join(os.path.dirname(__file__),"K.txt")).T

pcd_file = 'PointXYZI_009.pcd'
img_file = 'img_009.jpg'
pcd = o3d.io.read_point_cloud(pcd_file)
pcd_arr = np.array(pcd.points).T
img = np.array(Image.open(img_file))
img = cv2.undistort(img,ins,distort_params)
extran = np.loadtxt(os.path.join(os.path.dirname(__file__),sim_pose_file))
proj_pcd:np.ndarray = extran[:3,:3] @ pcd_arr + extran[:3,[3]]
proj_pcd = ins.dot(proj_pcd)
H,W = img.shape[:2]
u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
u = u/w
v = v/w
rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
u = u[rev]
v = v[rev]
r = 1/np.linalg.norm(pcd_arr[:,rev],axis=0)
plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
plt.axis([0,W,H,0])
plt.imshow(img)
plt.scatter([u],[v],c=[r],cmap='rainbow',alpha=0.8,s=3)
# plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__),'manual_gt.png'),bbox_inches='tight')
