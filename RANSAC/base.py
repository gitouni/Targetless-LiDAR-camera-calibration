import numpy as np

from scipy.spatial.transform import Rotation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error as mse

def vec_tran(rot_vec:np.ndarray):
    R = Rotation.from_rotvec(rot_vec)
    return R.as_matrix()

def toVec(Rmat:np.ndarray):
    R = Rotation.from_matrix(Rmat)
    vecR = R.as_rotvec()
    return vecR

def TL_solve(camera_edge,pcd_edge):
    assert len(camera_edge) == len(pcd_edge)
    N = len(camera_edge)
    alpha = np.zeros([N,3])
    beta = alpha.copy()
    for i,(cedge,pedge) in enumerate(zip(camera_edge,pcd_edge)):
        cvec = toVec(cedge)
        pvec = toVec(pedge)
        alpha[i,:] = cvec
        beta[i,:] = pvec
    H = np.dot(beta.T,alpha)  # (3,3)
    U, S, Vt = np.linalg.svd(H)  
    R = np.dot(Vt.T, U.T)     
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    return R, S, alpha, beta

class RotEstimator(BaseEstimator,RegressorMixin):
    def __init__(self,trans=np.eye(3)) -> None:
        self.trans:np.ndarray = trans
        
    @staticmethod
    def toVecList(mat_list:list):
        return np.array([toVec(mat) for mat in mat_list])
    
    def fit(self,beta:np.ndarray,alpha:np.ndarray):
        beta_:np.ndarray = beta - beta.mean(axis=0,keepdims=True)
        alpha_ :np.ndarray = alpha - alpha.mean(axis=0, keepdims=True)
        H = np.dot(beta_.T,alpha_)
        U, _ ,Vt = np.linalg.svd(H)
        R = np.dot(Vt.T,U.T)
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        self.trans = R
        return self
    def score(self,beta:np.ndarray,alpha:np.ndarray):
        err = self.trans.dot(beta.T) - alpha.T
        return err.mean()
    def predict(self,beta:np.ndarray):
        alpha_t = self.trans.dot(beta.T)
        return alpha_t.T
    def get_params(self, deep=True):
        return super().get_params(deep)
    def set_params(self, **params):
        return super().set_params(**params)
    

class RotRANSAC:
    """RANSAC_ICP: ransac regressor for point-point icp\n
    recommend to scale p0,p1 into unitcube first.
    """
    def __init__(self,estimator:BaseEstimator,
                 min_samples=3,max_trials=10000,
                 ransac_threshold=None,loss='squared_error',
                 stop_prob=0.999,
                 random_state=None):
        assert loss in ['squared_error','absolute_error'], f"unknown loss:{loss}"
        self.ransac_estimator = RANSACRegressor(estimator=estimator,
                                          min_samples=min_samples,
                                          residual_threshold=ransac_threshold,max_trials=max_trials,
                                          loss='squared_error',stop_probability=stop_prob,
                                          random_state=random_state)
    def reset(self):
        if hasattr(self.ransac_estimator,"estimator_"):  # reset state
            self.ransac_estimator.estimator_.set_params(trans=np.eye(4))   
        
    @staticmethod
    def toVecList(mat_list:list):
        return np.array([toVec(mat) for mat in mat_list])
        
    def fit(self,beta:np.ndarray,alpha:np.ndarray):
        self.ransac_estimator.fit(beta,alpha)
        return self.ransac_estimator.estimator_.get_params()['trans'], self.ransac_estimator.inlier_mask_
    
class TslEstimator(BaseEstimator,RegressorMixin):
    def __init__(self,R=np.eye(3),trans=np.zeros(3),scale=1) -> None:
        self.R = R
        self.trans:np.ndarray = trans
        self.scale = scale
    
    def get_params(self, deep=True):
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)
    
    @staticmethod
    def toMat(SE3_flatten:np.ndarray):
        if len(SE3_flatten.shape) == 1:
            SE3_flatten = SE3_flatten.reshape(1,-1)
        SE3_mat = SE3_flatten.copy().reshape(-1,4,4)
        R,t = SE3_mat[:,:3,:3], SE3_mat[:,:3,3]
        return R,t
    
    @staticmethod
    def flatten(mat_list:list):
        return np.vstack([mat.copy().reshape(-1) for mat in mat_list])
    
    def fit(self,A,B):
        RA,tA = self.toMat(A)
        _,tB = self.toMat(B)
        AA = []
        BB = []
        for Ra,ta,tb in zip(RA,tA,tB):
            AA.append(np.hstack((Ra-np.eye(3),ta[:,None])))
            BB.append(self.R.dot(tb[:,None]))
        AA, BB = np.vstack(AA), np.vstack(BB)  # (3,4) -> (3N,4); (3,1) -> (3N,1)
        sol:np.ndarray = np.linalg.inv(AA.T @ AA) @ AA.T @ BB
        sol = sol.copy().reshape(-1)
        self.trans = sol[:3]
        self.scale = sol[3]
        return self
    
    def predict(self,A:np.ndarray):
        X = np.eye(4)
        X[:3,:3] = self.R
        X[:3,3] = self.trans
        BB = []
        AA = A.copy().reshape(-1,4,4)
        AA[:,:3,3] *= self.scale
        Xinv = np.linalg.inv(X)
        for a in AA:
            b:np.ndarray = Xinv @ a @ X
            BB.append(b.copy().reshape(1,-1))
        BB = np.vstack(BB)
        return BB
    
    def score(self,A:np.ndarray,B:np.ndarray):
        AA = A.copy().reshape(-1,4,4)
        BB = B.copy().reshape(-1,4,4)
        BB[:,:3,3] *= self.scale
        X = np.eye(4)
        X[:3,:3] = self.R
        X[:3,3] = self.trans
        Err = 0
        for a,b in zip(AA,BB):
            err:np.ndarray = a @ X - X @ b
            Err += err.mean()
        return Err / len(AA)
        

class TslRANSAC:
    """RANSAC_ICP: ransac regressor for point-point icp\n
    recommend to scale p0,p1 into unitcube first.
    """
    def __init__(self,estimator:BaseEstimator,
                 min_samples=3,max_trials=10000,
                 ransac_threshold=None,loss='squared_error',
                 stop_prob=0.999,
                 random_state=None):
        assert loss in ['squared_error','absolute_error'], f"unknown loss:{loss}"
        self.ransac_estimator = RANSACRegressor(estimator=estimator,
                                          min_samples=min_samples,
                                          residual_threshold=ransac_threshold,max_trials=max_trials,
                                          loss='squared_error',stop_probability=stop_prob,
                                          random_state=random_state)
    def reset(self):
        if hasattr(self.ransac_estimator,"estimator_"):  # reset state
            self.ransac_estimator.estimator_.set_params(trans=np.eye(4))   
        
    @staticmethod
    def flatten(mat_list:list):
        return np.vstack([mat.copy().reshape(-1) for mat in mat_list])
        
    def fit(self,A:np.ndarray,B:np.ndarray)->np.ndarray:
        self.ransac_estimator.fit(A,B)
        params = self.ransac_estimator.estimator_.get_params()
        return params['trans'], params['scale'] ,self.ransac_estimator.inlier_mask_

     
if __name__ == "__main__":
    import sys,os
    BASE_DIR = os.getcwd()
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join("../",BASE_DIR))
    from problem import read_pcd_json, read_camera_json, poseToFulledge, EdgeLoss
    rot_estimator = RotEstimator()
    pcd_poses = read_pcd_json("/data2/ROS/ex_calib/res/building_rot4/ransac_optimized_topt.json")
    camera_poses = read_camera_json("/data2/ROS/ex_calib/res/building_rot4/sfm_data.json")
    pcd_edge = poseToFulledge(pcd_poses)
    camera_edge = poseToFulledge(camera_poses)
    camera_rotedge, pcd_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge,pcd_edge])
    TL_rotation, *_ = TL_solve(camera_rotedge,pcd_rotedge)
    print('TL_rotation:\n{}'.format(TL_rotation))
    alpha, beta = map(rot_estimator.toVecList,[camera_rotedge,pcd_rotedge])
    rot_estimator.fit(beta,alpha)
    print('Base_rotation:\n{}'.format(rot_estimator.get_params()['trans']))
    alpha_pred = rot_estimator.predict(beta)
    print("MSE Loss:{}".format(mse(alpha,alpha_pred)))
    ransac_rot_estimator = RotRANSAC(rot_estimator)
    ransac_rot, inlier_mask = ransac_rot_estimator.fit(beta,alpha)
    print('RANSAC rotation:\n{}'.format(ransac_rot))
    Loss = EdgeLoss(camera_edge,pcd_edge,np.eye(3))
    TL_tran, TL_scale = Loss.LSM(TL_rotation)
    print('TL Translation:{}, Scale:{}'.format(TL_tran,TL_scale))
    tsl_estimator = TslEstimator(ransac_rot)
    camera_flatten, pcd_flatten = map(tsl_estimator.flatten,[camera_edge,pcd_edge])
    tsl_estimator.fit(camera_flatten,pcd_flatten)
    base_params = tsl_estimator.get_params()
    pcd_flatten_pred = tsl_estimator.predict(camera_flatten)
    print("MSE Loss:{}".format(mse(pcd_flatten,pcd_flatten_pred)))
    print("Base Translation:{}, Scale:{}".format(base_params['trans'],base_params['scale']))
    ransac_tsl_estimator = TslRANSAC(tsl_estimator)
    ransac_tsl, ransac_scale, inlier_mask = ransac_tsl_estimator.fit(camera_flatten,pcd_flatten)
    pcd_flatten_pred = ransac_tsl_estimator.ransac_estimator.predict(camera_flatten)
    print('RANSAC Translation:{}, Scale:{}:'.format(ransac_tsl,ransac_scale))
    print("MSE Loss:{}".format(mse(pcd_flatten[inlier_mask],pcd_flatten_pred[inlier_mask])))    
    print((~inlier_mask).sum())                       