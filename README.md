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

Please ensure your OpenMVG has been installed propoerly before this step. SequentialSfM engine is suitable for this task empircally. To implement, please ensure you have properly installed this library and refer to `path/to/OpenMVG/build/software/SfM/SfM_SequentialPipeline.py`. After you finished this step, the resultant `sfm_data.json` can be found in the given directory with extrinsic matrices (poses to the world coordinate).

<details>
 <summary> If you are a beginner of OpenMVG or need troubleshooting</summary>
 
 If you are a beginner of OpenMVG, please unfold ths <detail> and follow our intructions and use the [doc/SfM_SequentialPipeline.py](doc/SfM_SequentialPipeline.py) file to implement SfM.
 
 To use it, you need to follow some simple steps:
 * replace all `path/to/OpenMVG` strings in the script with your specific OpenMVG installation directory.
 * give the intrinsic parameters of your camera. You may put it into a ASCII-coded file or just modify the `intrinsic` variable in the above python script. Fortunately, coarse instrinsic paramters are also OK, as SfM will estimate them simultaneously.
 * run the command `python Sfm_SequentialPipeline.py input_dir output_dir ins_file`. The `input_dir` is the directory containing all your raw images while the `output_dir` is the directory you designate to store all resultant files of OpenMVG. Note that if you choose to modify the `intrinsic` variable in the last step. Please just leave out the third argument `ins_file`.
 * After it finishes running, you will find a `sfm_data.json` file in `path/to/output_dir/reconstruction_sequential/`.
 
</details>

# Step3: Estimate initial LiDAR poses with RANSAC (RANReg mentioned in our paper)

Please ensure your Open3D has been installed properly in your python environment before this step.
