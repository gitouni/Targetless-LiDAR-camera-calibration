# Targetless LiDAR camera calibration
## Official Implementation of the paper "Targetless Extrinsic Calibration of Camera and Low-resolution 3D LiDAR"
![](doc/Abstract.jpg)

* preprint: [https://doi.org/10.36227/techrxiv.22155149](https://doi.org/10.36227/techrxiv.22155149)
* All codes will be available soon after the IEEE publication.
* All advices, citations, support will be acknowledged and appreciated.
* Feel free to propose any issues you met.

# Environment
* Ubuntu 18.04/20.04 or Windows 10
* Python 3.8 or later
* g++ 7 or later
* cmake 3.1.0 or later

# Dependence
* [OpenMVG](https://github.com/openMVG/openMVG)
* [OpenMVS](https://github.com/cdcseacave/openMVS)
* [Open3D](https://github.com/isl-org/Open3D)  (pip install open3d)

# Step 1: Prepare Image and LiDAR data

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
This preprocess pipeline is accelearted by OpenMP. When it finishes, you can use `./view xxx.cpd` to verify whether the pcd has been correctly filtered.

# Step 2: Estimate image poses using OpenMVG

Please ensure your OpenMVG has been installed propoerly before this step. SequentialSfM engine is suitable for this task empircally. To implement, please refer to `path/to/OpenMVG/build/software/SfM/SfM_SequentialPipeline.py`. After you finished this step, the resultant `sfm_data.json` can be found in the given directory with extrinsic matrices (poses to the world coordinate).

<details>
 <summary> If you are a beginner of OpenMVG or need troubleshooting</summary>
 
 If you are a beginner of OpenMVG, please follow our intructions to use the [doc/SfM_SequentialPipeline.py](doc/SfM_SequentialPipeline.py) file to implement SfM.
 
 To use it, you need to follow some simple steps:
 * replace all `path/to/OpenMVG` strings in the script with your specific OpenMVG installation directory.
 * give the intrinsic parameters of your camera. You may put it into an ASCII-coded file or just modify the `intrinsic` variable in the above python script. Fortunately, coarse instrinsic paramters are also OK, as SfM will estimate them simultaneously.
 * run the command `python Sfm_SequentialPipeline.py input_dir output_dir ins_file`. The `input_dir` is the directory containing all your raw images while the `output_dir` is the directory you designate to store all resultant files of OpenMVG. Note that if you choose to modify the `intrinsic` variable in the last step. Please just leave out the third argument `ins_file`.
 * After it finishes running, you will find a `sfm_data.json` file in `path/to/output_dir/reconstruction_sequential/`.
 
</details>

# Step3: Estimate initial LiDAR poses with RANSAC (RANReg mentioned in our paper)

Please ensure your Open3D has been installed properly in your python environment before this step.
```bash
python multiway_reg.py --basedir xxx --res_dir xxx --input_dr data/proc_pcd 
```
The above command will implement Multiway Registration using RANSAC. Here are some explanations to main args:
* basedir: basic name of resultant directories, just name it as you wish
* res_dir: resultant directory containing a Open3D PoseGraph
* input_dir: directory containing all the preprocessed pcd files

<details>
 <summary> Modification on other args</summary>
 * step: use 1 pcd file among every `step` files. Please change this argument carfully, as the pcd poses should keep consistent with the numebr of camera poses.
 * pose_graph: the name of the resultant PoseGraph. If you change this arg, please keep it constant in the following steps.
 * voxel_size: the downsampling voxel size. Smaller size leads to better registration performance as a pay for efficiency.
 * radius: radius to compute information matrix. A larger radius will consider a larger range of points into surface alignment.
 * ne_method: we implement normal estimation different from Open3D. `o3d` indicates using the original Open3D Normal Estimation, but it performs worse in low-resolution laser scans.
 
</details?
