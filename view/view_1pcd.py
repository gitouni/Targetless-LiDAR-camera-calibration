import os
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import time
def capture_img(vis):
    image = vis.capture_screen_float_buffer()
    plt.imsave(os.path.join(os.path.dirname(__file__),time.strftime("%H-%M-%S",time.localtime())+".png"),np.array(image))
    return False

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def change_background_to_white(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    return False

def color_pcd(pcd):
    pts = np.array(pcd.points)
    N = pts.shape[0]
    colors = np.zeros([N, 3])
    height_max = np.max(pts[:, 2])
    height_min = np.min(pts[:, 2])
    delta_c = abs(height_max - height_min) / (255 * 2)
    color_n = (pts[:,2] - height_min)/delta_c
    mask = color_n <= 255
    n = mask.sum()
    color_n1 = 1-color_n[mask]/255  # (n,)
    color_n2 = color_n[~mask]/255 -1  # (N-n,)
    colors[mask,:] = np.hstack([np.zeros([n,1]),1-color_n1[:,None],color_n1[:,None],])
    colors[~mask,:] = np.hstack([color_n2[:,None],1-color_n2[:,None],np.zeros([N-n,1])])
    # for j in range(pts.shape[0]):
    #     color_n = (pts[j, 2] - height_min) / delta_c
    #     if color_n <= 255:
    #         colors[j, :] = [0, 1 - color_n / 255, 1]
    #     else:
    #         colors[j, :] = [(color_n - 255) / 255, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

if __name__ == "__main__":
    pcd_path = "/data2/ROS/ex_calib/res/building_review2/floam_recon.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    # pose_graph = o3d.io.read_pose_graph("/data2/ROS/ex_calib/res/building_rot4/ransac_optimized_defined.json")
    # invpose = np.linalg.inv(pose_graph.nodes[32].pose)
    # pcd.transform(invpose)
    pcd = color_pcd(pcd)
    # colors = np.array(pcd.colors)[None,...]
    # colors = (colors*255).astype(np.uint8)
    # hsv = cv2.cvtColor(colors,cv2.COLOR_RGB2HSV)
    # hsv = hsv[0]
    # remove_idx = (hsv[:,0]>72)*(hsv[:,0]<108)
    # idx = np.arange(hsv.shape[0])
    # pcd = pcd.select_by_index(idx[remove_idx],invert=True)
    key_to_callback = {}
    key_to_callback[ord(".")] = capture_img
    o3d.visualization.draw_geometries_with_key_callbacks([pcd],key_to_callback)