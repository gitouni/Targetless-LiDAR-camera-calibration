#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python implementation of the bash script written by Romuald Perrot
# Created by @vins31
# Modified by Pierre Moulon
#
# this script is for easy use of OpenMVG
#
# usage : python openmvg.py image_dir output_dir
#
# image_dir is the input directory where images are located
# output_dir is where the project must be saved
#
# if output_dir is not present script will create it
#
# build/Linux-x86_64-RELEASE
# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/home/bit/CODE/Research/AutoCalib/openMVG/build/Linux-x86_64-RELEASE"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/bit/CODE/Research/AutoCalib/openMVG/src/openMVG/exif/sensor_width_database"

import os
import subprocess
import numpy as np
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("input_dir",type=str)
parser.add_argument("output_dir",type=str)
parser.add_argument("--intrinsic_file",dest="ins_file",type=str,default=None)
parser.add_argument("--remove_indice",type=int,nargs="+",default=[])
args = parser.parse_args()

os.chdir(os.path.abspath(os.path.dirname(__file__)))
input_dir = args.input_dir
output_dir = args.output_dir
if args.ins_file is not None:
  ins_dir = args.ins_file
  if os.path.exists(ins_dir):
    ins:np.ndarray = np.loadtxt(ins_dir,dtype=np.float32)
    ins = [str(num.item()) for num in ins.flatten()]
    intrinsic = ";".join(ins)
    print('Read intrinsic from file:\n{}'.format(intrinsic))
else:
  print('No intrinsic dir, use default instrinsic')
  intrinsic = "1069.053955078125;0.0;637.1859130859375;0.0;1068.8653564453125;489.39117431640625;0.0;0.0;1.0"
matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)
if len(args.remove_indice) > 0:
  input_tmp_dir = os.path.join(os.path.dirname(input_dir),"tmp_img")
  if os.path.exists(input_tmp_dir):
    print("Existing tmp directory {} will be removed.".format(input_tmp_dir))
    shutil.rmtree(input_tmp_dir)
  os.makedirs(input_tmp_dir)
  img_files = list(sorted(os.listdir(input_dir)))
  for i,filename in enumerate(img_files):
    if i in args.remove_indice:
      continue
    os.system('ln -s {} {}'.format(os.path.join(input_dir,filename),os.path.join(input_tmp_dir,filename)))
  input_dir = input_tmp_dir
  print("Input dir changed to {}".format(input_tmp_dir))
# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)

print ("1. Intrinsics analysis")
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-k", intrinsic] )
pIntrisics.wait()

print ("2. Compute features")
pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m",  "SIFT"] )
pFeatures.wait()

print ("3. Compute matching pairs")
pPairs = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"), "-i", matches_dir+"/sfm_data.json", "-o" , matches_dir + "/pairs.bin" ] )
pPairs.wait()

print ("4. Compute matches")
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-p", matches_dir+ "/pairs.bin", "-o", matches_dir + "/matches.putative.bin" ] )
pMatches.wait()

print ("5. Filter matches" )
pFiltering = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir+"/sfm_data.json", "-m", matches_dir+"/matches.putative.bin" , "-g" , "f" , "-o" , matches_dir+"/matches.f.bin" ] )
pFiltering.wait()

# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)

print ("6. Do Sequential/Incremental reconstruction")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", "INCREMENTALV2", "--input_file", matches_dir+"/sfm_data.json", "--match_dir", matches_dir, "--output_dir", reconstruction_dir] )
pRecons.wait()

print ("7. Colorize Structure")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
pRecons.wait()

print ("8. Convert data format")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i", reconstruction_dir+"/sfm_data.bin", "-o", reconstruction_dir+"/sfm_data.json", "-V", "-I", "-E"])
pRecons.wait()

print ("9 Export undistorted images")
pRecons = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ExportUndistortedImages"),'-i', matches_dir+"/sfm_data.json",'-o',reconstruction_dir+"/undistorted_images"])
pRecons.wait()

print ("10 Output OpenMVS")
pRecons = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2openMVS"), "-i", reconstruction_dir+"/sfm_data.bin", "-o", reconstruction_dir+"/scene.mvs"])
pRecons.wait()
