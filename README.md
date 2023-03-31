# Targetless LiDAR camera calibration
Official Implementation of the paper "Targetless Extrinsic Calibration of Camera and Low-resolution 3D LiDAR"
* preprint: [https://doi.org/10.36227/techrxiv.22155149](https://doi.org/10.36227/techrxiv.22155149)
* All codes will be available soon after the publication of this paper.

# Environment
* Ubuntu 18.04/20.04 or Windows 10
* Python 3.8 or later
* g++ 7 or later
* cmake 3.20.0 or later

# Dependence
* [OpenMVG](https://github.com/openMVG/openMVG)
* [OpenMVS](https://github.com/cdcseacave/openMVS)
* [Open3D](https://github.com/isl-org/Open3D)  (pip install open3d)

# Step 1: Prepare Image and LiDAR data:

You need to add sychronized image and LiDAR data to the respective directories first.
```bash
mkdir data && cd data
mkdir img
mkdir proc
```

Use our cpp tools to preprocess LiDAR data. (Remove the backward 180 degrees of each LiDAR scan)

```bash
cd cpp
mkdir build && cd build
cmake ..
make
./preprocess ../../data/pcd ../../data/proc_pcd
```


 
