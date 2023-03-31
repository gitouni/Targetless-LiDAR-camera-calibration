import torch
import numpy as np
from FCGF.model.resunet import ResUNetBN2C
import os
import MinkowskiEngine as ME


def get_fcgf_model(path:str='kitti_v0.3.pth'):
    model_path = os.path.join(os.path.dirname(__file__),path)
    model = ResUNetBN2C(in_channels=1,out_channels=32,conv1_kernel_size=5,normalize_feature=True)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    return model

def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')

def find_knn_gpu(F0, F1, nn_max_n=-1, knn=1, return_distance=False):

    def knn_dist(f0, f1, knn=1, dist_type='L2'):
        knn_dists, knn_inds = [], []
        with torch.no_grad():
            dist = pdist(f0, f1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1, keepdim=True)

            knn_dists.append(min_dist)
            knn_inds.append(ind)

            if knn > 1:
                for _ in range(knn - 1):
                    NR, NC = dist.shape
                    flat_ind = (torch.arange(NR) * NC).type_as(ind) + ind.squeeze()
                    dist.view(-1)[flat_ind] = np.inf
                    min_dist, ind = dist.min(dim=1, keepdim=True)

                    knn_dists.append(min_dist)
                    knn_inds.append(ind)

                min_dist = torch.cat(knn_dists, 1)
                ind = torch.cat(knn_inds, 1)

                return min_dist, ind

            # Too much memory if F0 or F1 large. Divide the F0
            if nn_max_n > 1:
                N = len(F0)
                C = int(np.ceil(N / nn_max_n))
                stride = nn_max_n
                dists, inds = [], []
                with torch.no_grad():
                    for i in range(C):
                        dist, ind = knn_dist(F0[i * stride:(i + 1) * stride], F1, knn=knn, dist_type='L2')
                        dists.append(dist)
                        inds.append(ind)

                dists = torch.cat(dists)
                inds = torch.cat(inds)
                assert len(inds) == N

            else:
                dist = pdist(F0, F1, dist_type='SquareL2')
                min_dist, inds = dist.min(dim=1)
                dists = min_dist.detach().unsqueeze(1) #.cpu()
                # inds = inds.cpu()
            if return_distance:
                return inds, dists
            else:
                return inds

class FCGF_Extractor:
    def __init__(self,model_path:str='kitti_v0.3.pth',device='cuda:0'):
        self.device = device
        self.model_path = model_path
        self.model = get_fcgf_model(model_path).to(device)
        self.model.eval()
        self.quantize_size = 0.3
    def extract_fcgf(self,pcd:np.ndarray):
        if pcd.shape[0] == 3:
            _pcd = torch.IntTensor(pcd.transpose(0,1)/self.quantize_size)
        else:
            _pcd = torch.IntTensor(pcd/self.quantize_size)
        quantized_pcd, _, inv_mapping = ME.utils.sparse_quantize(_pcd.contiguous(),return_index=True,return_inverse=True)
        feats = torch.ones(quantized_pcd.shape[0],1,dtype=torch.float)
        batched_coords = ME.utils.batched_coordinates([quantized_pcd],device=self.device)
        sinput = ME.SparseTensor(coordinates=batched_coords,features=feats,device=self.device)
        soutput = self.model(sinput)
        out_feat:torch.Tensor = soutput.features_at(0)[inv_mapping]
        return out_feat.detach().cpu().numpy()  # (N,D)
        


    