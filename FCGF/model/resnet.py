import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm
from model.residual_block import get_block


class ResNet(ME.MinkowskiNetwork):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128]
    REGION_TYPE = ME.RegionType.HYPER_CUBE

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                in_channels=1,
                bn_momentum=0.1,
                conv1_kernel_size=3,
                normalize_feature=False,
                D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        REGION_TYPE = self.REGION_TYPE
        self.normalize_feature = normalize_feature

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)


    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        return out
    
def preprocess(xyz, voxel_size=0.3):
    '''
    Stage 0: preprocess raw input point cloud
    Input: raw point cloud
    Output: voxelized point cloud with
    - xyz:    unique point cloud with one point per voxel
    - coords: coords after voxelization
    - feats:  dummy feature placeholder for general sparse convolution
    '''

    # Voxelization:
    # Maintain double type for xyz to improve numerical accuracy in quantization
    coords,sel = ME.utils.sparse_quantize(xyz / voxel_size, return_index=False)
    npts = len(sel)

    xyz = torch.from_numpy(xyz)

    # ME standard batch coordinates
    # coords = ME.utils.batched_coordinates([torch.floor(xyz / voxel_size).int()])
    feats = torch.ones(npts, 1)

    return xyz.float(), coords, feats

    
    