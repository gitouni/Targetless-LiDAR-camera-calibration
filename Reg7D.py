import open3d as o3d
import numpy as np
import json
import os
from copy import deepcopy
from matplotlib import pyplot as plt
import time
from clique_utils import flatten_clique_list
import argparse

work_dir = "building_imu"
method = "ranreg"
def input_args():
    parser = argparse.ArgumentParser()
    load_parser = parser.add_argument_group()
    load_parser.add_argument("--work_dir",type=str,default="res/{}/".format(work_dir))
    load_parser.add_argument("--lidar_pcd",type=str,default="{}_union.pcd".format(method))
    load_parser.add_argument("--camera_pcd",type=str,default="scene_dense.ply")
    load_parser.add_argument("--camera_json",type=str,default="sfm_data.json")
    load_parser.add_argument("--TL_init",type=str,default="TL_{}_fsol.npz".format(method))
    load_parser.add_argument("--clique_file",type=str,default="clique_{}.json".format(method))
    save_parser = parser.add_argument_group()
    save_parser.add_argument("--save_dir",type=str,default="res/{}/".format(work_dir))
    save_parser.add_argument("--save_tcl",type=str,default="{}_tcl.txt".format(method))
    save_parser.add_argument("--save_7dof",type=str,default="{}_7dof.txt".format(method))
    save_parser.add_argument("--save_6dof",type=str,default="{}_6dof.txt".format(method))
    args = parser.parse_args()
    args.lidar_pcd, args.camera_pcd, args.camera_json, args.TL_init, args.clique_file = map(lambda filename: os.path.join(args.work_dir,filename),
        [args.lidar_pcd, args.camera_pcd, args.camera_json, args.TL_init, args.clique_file])
    args.save_tcl, args.save_7dof, args.save_6dof = map(lambda filename: os.path.join(args.save_dir,filename),
        [args.save_tcl, args.save_7dof, args.save_6dof])
    return args

def compute_fpfh(pcd:o3d.geometry.PointCloud):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(0.9,30)
        )
    return pcd_fpfh


def capture_img(vis):
    image = vis.capture_screen_float_buffer()
    plt.imsave(os.path.join('view',time.strftime("%H-%M-%S",time.localtime())+".png"),np.array(image))
    return False


if __name__ == "__main__":
    args = input_args()
    pcd1_filename = args.camera_pcd
    pcd2_filename = args.lidar_pcd
    camera_json = args.camera_json
    rpcd1 = o3d.io.read_point_cloud(pcd1_filename)
    pcd2 = o3d.io.read_point_cloud(pcd2_filename)
    pcd1 = deepcopy(rpcd1)
    with open(camera_json,'r')as f:
        data = json.load(f)
    cliques = json.load(open(args.clique_file,'r'))
    cliques = flatten_clique_list(cliques)
    print('Inliers index:\n{}'.format(cliques))
    camera_extran0 = data['extrinsics'][cliques[0]]  # first index of clique list
    T = np.eye(4)
    R = np.array(camera_extran0['value']['rotation'])
    t = np.array(camera_extran0['value']['center'])
    T[:3,:3] = R
    T[:3,3] = -R @ t
    pcd1.transform(T)
    ex_init = np.load(args.TL_init)
    extran0 = np.eye(4)
    extran0[:3,:3] = ex_init['rotation']
    extran0[:3,3] = ex_init['translation']
    print('TCL:\n{}'.format(extran0))
    extran0 = np.linalg.inv(extran0)  # TCL -> TLC
    np.savetxt(args.save_tcl,extran0)

    extran0[:3,:3] *= ex_init['scale']
    pcd1.transform(extran0)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

    key_to_callback = {}
    key_to_callback[ord(".")] = capture_img
    # o3d.visualization.draw_geometries_with_key_callbacks([pcd1,pcd2],key_to_callback)
    o3d.io.write_point_cloud('tmp/camera_tcl.pcd',pcd1)
    o3d.io.write_point_cloud('tmp/pcd_tcl.pcd',pcd2)

    print(len(pcd1.points),len(pcd2.points))



    res = o3d.pipelines.registration.registration_icp(
                        pcd1,pcd2,0.3,np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                    )
    tmp = res.transformation

    pcd1.transform(tmp)
    o3d.io.write_point_cloud('tmp/camera_7dof.pcd',pcd1)
    o3d.io.write_point_cloud('tmp/pcd_7dof.pcd',pcd2)
    # np.savetxt(args.save_refinement,tmp)
    res_tmp = tmp.dot(extran0)
    np.savetxt(args.save_7dof,res_tmp)
    RS = res_tmp[:3,:3] @ res_tmp[:3,:3].T
    s = np.sqrt(RS[0,0])
    res_tmp[:3,:3] /= s
    print('Scale:{}'.format(s))
    np.savetxt(args.save_6dof,res_tmp)  # Transformation from camera to LiDAR
    print('Final transform:{}'.format(res_tmp))
    o3d.visualization.draw_geometries_with_key_callbacks([pcd1,pcd2],key_to_callback)
