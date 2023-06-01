import open3d as o3d
import numpy as np

def compute_fpfh(pcd:o3d.geometry.PointCloud,n:int=30):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamKNN(n)
        )
    return pcd_fpfh

def register(pcd1,pcd2,normal_radius=0.3,n=30,icp_tuned=False,estimate_normal=False,max_iter=100000,return_fitness=False):
    if estimate_normal:
        pcd1.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius,max_nn=30))
        pcd2.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius,max_nn=30))
    pcd1_fpfh = compute_fpfh(pcd1,n)
    pcd2_fpfh = compute_fpfh(pcd2,n)
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1,pcd2,pcd1_fpfh,pcd2_fpfh,True,0.6,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    0.6)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999))

    tmp = res.transformation
    if icp_tuned:
        res = o3d.pipelines.registration.registration_icp(
                    pcd1,pcd2,0.05,tmp,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
        tmp = res.transformation
    if not return_fitness:
        return tmp
    else:
        return tmp, res.fitness

def register_with_feature(pcd1,pcd2,pcd_feat1,pcd_feat2,max_iter=100000,seed=None,method="point2point",icp_tuned=False,return_fitness=False):
    methods = {"point2point":o3d.pipelines.registration.TransformationEstimationPointToPoint(),
               "point2plane":o3d.pipelines.registration.TransformationEstimationPointToPlane()}
    icp_method = methods[method]
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1,pcd2,pcd_feat1,pcd_feat2,True,0.6,
                                icp_method,
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    0.3)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999),seed=seed)

    tmp = res.transformation
    if icp_tuned:
        res = o3d.pipelines.registration.registration_icp(
                    pcd1,pcd2,0.05,tmp,
                    icp_method
                )
        tmp = res.transformation
    if not return_fitness:
        return tmp
    else:
        return tmp, res.fitness
    
def register_with_feature2(pcd1,pcd2,pcd_feat1,pcd_feat2,max_iter=10000,method="point2point",icp_tuned=False,return_corr=False):
    methods = {"point2point":o3d.pipelines.registration.TransformationEstimationPointToPoint(),
               "point2plane":o3d.pipelines.registration.TransformationEstimationPointToPlane()}
    icp_method = methods[method]
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1,pcd2,pcd_feat1,pcd_feat2,True,0.6,
                                icp_method,
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    0.3)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999))

    tmp = res.transformation
    if icp_tuned:
        res = o3d.pipelines.registration.registration_icp(
                    pcd1,pcd2,0.05,tmp,
                    icp_method
                )
        tmp = res.transformation
    if not return_corr:
        return tmp
    else:
        return tmp, np.array(res.correspondence_set)
    