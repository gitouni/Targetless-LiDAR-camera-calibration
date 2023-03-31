from functools import partial
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from joblib import Parallel,delayed
from matplotlib import pyplot as plt

def ComputeEigenvector0(A:np.ndarray, eval0:float):
    row = A - np.diag(eval0*np.ones(3))
    rxr = np.array([np.cross(row[0],row[1]),np.cross(row[0],row[2]),np.cross(row[1],row[2])])
    d = np.sum(rxr**2,axis=1)
    imax = np.argmax(d)
    dmax = d[imax]
    return rxr[imax]/np.sqrt(dmax)
    
def ComputeEigenvector1(A:np.ndarray, evec0:np.ndarray, eval1:float):
    if abs(evec0[0]) > abs(evec0[1]):
        inv_length = 1/np.sqrt(evec0[0]**2+evec0[2]**2)
        U = np.array([-evec0[2]*inv_length ,0, evec0[0]*inv_length])
    else:
        inv_length = 1/np.sqrt(evec0[1]**2+evec0[2]**2)
        U = np.array([0, evec0[2]*inv_length, -evec0[1]*inv_length])
    V = np.cross(evec0,U)
    AU = np.sum(A*U[np.newaxis,:],axis=1)
    AV = np.sum(A*V[np.newaxis,:],axis=1)
    m00 = np.dot(U,AU) - eval1
    m01 = np.dot(U,AV)
    m11 = np.dot(V,AV) - eval1
    absM00, absM01, absM11 = map(abs,[m00,m01,m11])
    if absM00 >= absM11:
        max_abs_comp = max(absM00, absM01)
        if max_abs_comp > 0:
            if absM00 >= absM01:
                m01 /= m00
                m00 = 1 / np.sqrt(1 + m01 * m01)
                m01 *= m00
            else:
                m00 /= m01
                m01 = 1 / np.sqrt(1 + m00 * m00)
                m00 *= m01
            
            return m01 * U - m00 * V
        else:
            return U
    else:
        max_abs_comp = max(absM11, absM01)
        if max_abs_comp > 0:
            if absM11 >= absM01:
                m01 /= m11
                m11 = 1 / np.sqrt(1 + m01 * m01)
                m01 *= m11
            else:
                m11 /= m01
                m01 = 1 / np.sqrt(1 + m11 * m11)
                m11 *= m01
            
            return m11 * U - m01 * V
        else:
            return U
    
def FastEigen3x3(sub_pcd:np.ndarray):
    if sub_pcd.size == 0:
        A = np.identity(3)
    elif sub_pcd.shape[0] < 3:
        A = np.identity(3)
    else:
        A:np.ndarray = np.cov(sub_pcd.T)  # (3,3)
    Amax = A.max()
    if Amax == 0:
        return np.zeros(3)
    else:
        A /= Amax
    norm = A[0,1]**2 + A[0,2]**2 + A[1,2]**2
    eval_ = np.zeros(3)
    beta = np.zeros(3)
    if norm > 0:
        q = np.trace(A)/3
        b00 = A[0,0] - q
        b11 = A[1,1] - q
        b22 = A[2,2] - q
        p = np.sqrt((b00**2+b11**2+b22**2+norm*2)/6)
        c00 = b11*b22 - A[1,2]**2
        c01 = A[0,1]*b22 - A[1,2]*A[0,2]
        c02 = A[0,1]*A[1,2] - b11*A[0,2]
        det = (b00*c00 - A[0,1]*c01 + A[0,2]*c02)/p**3
        half_det = np.clip(det/2,-1,1)
        angle = np.arccos(half_det)/3
        beta[2] = np.cos(angle)*2
        beta[0] = np.cos(angle+2/3*np.pi)*2
        beta[1] = -(beta[0] + beta[2])
        eval_ = q + p*beta
        if half_det >= 0:
            evec2 = ComputeEigenvector0(A, eval_[2])
            if eval_[2] < eval_[0] and eval_[2] < eval_[1]:
                A *= Amax
                return evec2
            evec1 = ComputeEigenvector1(A, evec2, eval_[1])
            A *= Amax
            if eval_[1] < eval_[0] and eval_[1] < eval_[2]:
                return evec1
            evec0 = np.cross(evec1, evec2)
            return evec0
        else:
            evec0 = ComputeEigenvector0(A, eval_[0])
            if eval_[0] < eval_[1] and eval_[0] < eval_[2]:
                A *= Amax
                return evec0
            evec1 = ComputeEigenvector1(A, evec0, eval_[1])
            A *= Amax
            if eval_[1] < eval_[0] and eval_[1] < eval_[2]:
                return evec1
            evec2 = np.cross(evec0, evec1)
            return evec2
    else:
        A *= Amax
        if A[0,0] < A[1,1] and A[0,0] < A[2,2]:
            return np.array([1,0,0])
        elif A[1,1] < A[0,0] and A[1,1] < A[2,2]:
            return np.array([0,1,0])
        else:
            return np.array([0,0,1])    

def coords_tran(pcd_arr:np.ndarray):
    x,y,z = pcd_arr[:,0], pcd_arr[:,1], pcd_arr[:,2]
    d = np.sqrt(x**2 + y**2)
    angle = np.arctan2(z,np.sqrt(x**2+y**2))
    yaw = np.arctan2(y,x)
    return np.hstack([d[:,np.newaxis],5*angle[:,np.newaxis],20*yaw[:,np.newaxis]])


def pcd_coords_tran(pcd:o3d.geometry.PointCloud):
    pcd_arr = np.array(pcd.points)
    coords = coords_tran(pcd_arr)
    tran_pcd = o3d.geometry.PointCloud()
    tran_pcd.points = o3d.utility.Vector3dVector(coords)
    return tran_pcd

def capture_img(vis,save_path:str):
    image = vis.capture_screen_float_buffer()
    plt.imsave(save_path,np.array(image))
    return False
    

class Surface_Normal:
    def __init__(self,radius=0.3,n_neighbors=5,workers=1):
        # concat: 'none','xyz' or 'zero-mean'
        self.radius = radius
        self.n_eighbors = n_neighbors
        self.workers = workers
        
    def __call__(self, pcd: np.ndarray):
        kdtree = cKDTree(pcd,leafsize=30)
        _, query = kdtree.query(pcd,k=self.n_eighbors,workers=-1)
        parellel = Parallel(self.workers)
        pcd_norm = parellel(delayed(FastEigen3x3)(pcd[index]) for index in query)
        return np.array(pcd_norm)
    
class Surface_TNormal:
    def __init__(self,radius=0.3,n_neighbors=5,workers=-1,method='hybrid'):
        # concat: 'none','xyz' or 'zero-mean'
        method = method.lower()
        assert method in ['hybrid','knn'], 'Invalid search type :{}'.format(method)
        self.radius = radius
        self.n_eighbors = n_neighbors
        self.workers = workers
        self.method = method
        
    def __call__(self, pcd:np.ndarray):
        coords = coords_tran(pcd)
        query = self.search_query(coords,30)
        parellel = Parallel(self.workers,batch_size=200)
        pcd_norm = parellel(delayed(FastEigen3x3)(pcd[index]) for index in query)
        return np.array(pcd_norm)
    
    def search_query(self,coords:np.ndarray, leaf_size=30):
        kdtree = cKDTree(coords,leafsize=leaf_size)
        if self.method == 'knn':
            _, query = kdtree.query(coords,k=self.n_eighbors,workers=-1)
            return query
        else:
            dist, query = kdtree.query(coords,k=self.n_eighbors,workers=-1)  # N,K
            dist_rev = dist <= self.radius
            query_list = [q[rev] for q,rev in zip(query,dist_rev)]
        return query_list
    def estimate_normals(self,pcd:o3d.geometry.PointCloud):
        pcd_arr = np.array(pcd.points)
        pcd_norm = self(pcd_arr)
        pcd.normals = o3d.utility.Vector3dVector(pcd_norm)
        return pcd

def resize_normal_length(pcd:o3d.geometry.PointCloud,resize_ratio=0.2):
    pcd_norm = np.array(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(resize_ratio*pcd_norm)
    return pcd

if __name__ == "__main__":
    pcd_filename = '/data2/ROS/ex_calib/building_rot/proc_pcd/PointXYZI_001.pcd'
    save_path = '/data2/ROS/ex_calib/view/demo_metric.png'
    key_to_callback = {}
    
    # key_to_callback[ord("n")] = show_normal
    key_to_callback[ord(".")] = partial(capture_img,save_path=save_path)
    defined_normal = True
    display_coords = True
    knn_num = 30
    radius = 0.5
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    pcd = o3d.io.read_point_cloud(pcd_filename)
    o3d_knn_method = o3d.geometry.KDTreeSearchParamHybrid(radius,knn_num)
    
    if defined_normal:
        if not display_coords:
            pcd_arr = np.array(pcd.points)
            
            ne = Surface_TNormal(n_neighbors=knn_num,workers=-1)
            pcd = ne.estimate_normals(pcd)
            pcd = resize_normal_length(pcd)
            o3d.visualization.draw_geometries_with_key_callbacks([pcd,origin],key_to_callback)
        else:
            coord_pcd = pcd_coords_tran(pcd)
            coord_pcd.estimate_normals(o3d_knn_method)
            coord_pcd = resize_normal_length(coord_pcd)
            o3d.visualization.draw_geometries_with_key_callbacks([coord_pcd,origin],key_to_callback)
    else:
        pcd.estimate_normals(
            o3d_knn_method)
        pcd = resize_normal_length(pcd)
        o3d.visualization.draw_geometries_with_key_callbacks([pcd,origin],key_to_callback)
# pcd.estimate_normals(
#         o3d.geometry.KDTreeSearchParamKNN())
# pcd_norm = np.array(pcd.normals)
# print(pcd_norm[:10])
# pcd.paint_uniform_color([1,0.7,0])



