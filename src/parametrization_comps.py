import torch

from torch import nn
from torch import functional as F

from utils import *


class NbrAgg(nn.Module):
    def __init__(self, num_neighbors, out_channels):
        super(NbrAgg, self).__init__()
        self.num_neighbors = num_neighbors
        self.out_channels = out_channels
        self.smlp_1 = nn.Sequential(SMLP(7, 16, True, 'relu'), SMLP(16, out_channels, True, 'relu'))
        self.smlp_2 = SMLP(3, out_channels, True, 'relu')
        self.smlp_3 = SMLP(out_channels*2, out_channels, True, 'relu')

    def forward(self, pts: torch.Tensor):
        assert pts.ndim == 3 
        assert pts.size(2) == 3
        B, N, K, C = pts.size(0), pts.size(1), self.num_neighbors, self.out_channels
        knn_idx = knn_search(pts, pts, K+1)
        knn_pts = index_points(pts, knn_idx)
        abs_pts = knn_pts[:, :, :1, :]
        rel_nbs = knn_pts[:, :, 1:, :] - knn_pts[:, :, :1, :]
        dists = torch.sqrt((rel_nbs ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        concat = torch.cat((abs_pts.repeat(1, 1, K, 1), rel_nbs, dists), dim=-1)
        nbs_pooled = self.smlp_1(concat.view(B*N, K, -1)).view(B, N, K, -1).max(dim=2)[0]
        pts_lifted = self.smlp_2(pts)
        pts_ebd = self.smlp_3(torch.cat((pts_lifted, nbs_pooled), dim=-1))
        return pts_ebd


class AttPool(nn.Module):
    def __init__(self, in_chs):
        super(AttPool, self).__init__()
        self.in_chs = in_chs
        self.linear_transform = SMLP(in_chs, in_chs, False, 'none')

    def forward(self, x: torch.Tensor):
        assert x.ndim==3 and x.size(2)==self.in_chs
        # NOTE: This may be a gentle approach to maximum pooling.
        scores = F.softmax(self.linear_transform(x), dim=1)
        y = (x * scores).sum(dim=1)
        return y


class CdwExtractor(nn.Module):
    def __init__(self):
        super(CdwExtractor, self).__init__()
        # NOTE: A "sample+query(knn)" based point embedding.
        self.loc_agg = NbrAgg(16, 32)
        self.res_smlp_1 = ResSMLP(32, 64)
        self.res_smlp_2 = ResSMLP(128, 128)
        self.fuse = SMLP(352, 512, True, 'relu')
        self.att_pool = AttPool(512)
        self.fc = nn.Sequential(FC(1024, 512, True, 'relu'), FC(512, 1024, True, 'relu'), FC(1024, 1024, False, 'none'))

    def forward(self, pts: torch.Tensor):
        _, N, _ = pts.size()
        # NOTE: Style of this extractor looks like DGCNN due to
        # its skip concatenations and pooling(attention) operation.
        ftr_1 = self.loc_agg(pts)
        ftr_2 = self.res_smlp_1(ftr_1)
        ftr_3 = self.res_smlp_2(torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        # NOTE: Fuse all features extracted in previous stages.
        ftr_4 = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3, ftr_3.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        cdw = self.fc(torch.cat((ftr_4.max(dim=1)[0], self.att_pool(ftr_4)), dim=-1))
        return cdw  # (B,N,3) -> (B,1,1024)


class G2SD(nn.Module):
    def __init__(self, num_grids):
        super(G2SD, self).__init__()
        # NOTE: Create a lattice ('num_grids' points distributed uniformly in a square with unit length)
        self.num_grids = num_grids
        self.grid_size = int(np.sqrt(num_grids))
        assert self.grid_size**2 == self.num_grids
        self.lattice = torch.tensor(build_lattice(self.grid_size, self.grid_size)[0])
        # NOTE: Encoder & MLPs (two-stack learning architecture).
        self.backbone = CdwExtractor()  # Encoder
        fold_1_1 = SMLP(1026, 256, True, 'relu')
        fold_1_2 = SMLP(256, 128, True, 'relu')
        fold_1_3 = SMLP(128, 64, True, 'relu')
        fold_1_4 = SMLP(64, 3, False, 'none')
        self.fold_1 = nn.Sequential(fold_1_1, fold_1_2, fold_1_3, fold_1_4)
        fold_2_1 = SMLP(1027, 256, True, 'relu')
        fold_2_2 = SMLP(256, 128, True, 'relu')
        fold_2_3 = SMLP(128, 64, True, 'relu')
        fold_2_4 = SMLP(64, 3, False, 'none')
        self.fold_2 = nn.Sequential(fold_2_1, fold_2_2, fold_2_3, fold_2_4)

    def forward(self, pts: torch.Tensor):
        B, N, device = pts.size(0), pts.size(1), pts.device
        grids = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)  # (B,N',2)
        # NOTE: Encode points(B,N,3) into a codeword(B,1,d{1024}).
        cdw: torch.Tensor = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)  # (B,N',1024)
        # NOTE: Apply MLPs to grids or reconstructed points with duplicated codeword.
        concat_1 = torch.cat((cdw_dup, grids), dim=-1)
        rec_1 = self.fold_1(concat_1)  # (B,N',1026{1024+2}) -> (B,N',3)
        concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
        rec_2 = self.fold_2(concat_2)  # (B,N',1027{1024+3}) -> (B,N',3)
        return rec_2  # Reconstructed points with size (B,N'{N},3).


class PatCdwExtractor(nn.Module):
    def __init__(self):
        super(PatCdwExtractor, self).__init__()
        self.lift = SMLP(3, 16, True, 'relu')
        self.res_smlp_1 = ResSMLP(16, 32)
        self.res_smlp_2 = ResSMLP(64, 64)
        self.fuse = SMLP(176, 128, True, 'relu')
        self.att_pool = AttPool(128)
        self.fc = nn.Sequential(FC(256, 128, True, 'relu'), FC(128, 128, True, 'relu'), FC(128, 128, False, 'none'))

    def forward(self, pts: torch.Tensor):
        _, N, _ = pts.size()
        # NOTE: Style of this extractor looks like DGCNN due to
        # its skip concatenations and pooling(attention) operation.
        ftr_1 = self.lift(pts)  # (B,N,16)
        ftr_2 = self.res_smlp_1(ftr_1)  # (B,N,32)
        ftr_3 = self.res_smlp_2(torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        # NOTE: Fuse all features extracted in previous stages.
        ftr_4 = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3, ftr_3.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        cdw = self.fc(torch.cat((ftr_4.max(dim=1)[0], self.att_pool(ftr_4)), dim=-1))
        return cdw  # (B,N,3) -> (B,1,128)


class S2PF(nn.Module):
    def __init__(self, rescale_ebd=True):
        super(S2PF, self).__init__()
        # NOTE: Encoder & MLPs.
        self.backbone = PatCdwExtractor()  # Encoder
        unfold_1_1 = SMLP(131, 128, True, 'relu')
        unfold_1_2 = SMLP(128, 128, True, 'relu')
        unfold_1_3 = SMLP(128, 64, True, 'relu')
        unfold_1_4 = SMLP(64, 2, False, 'none')
        self.unfold_1 = nn.Sequential(unfold_1_1, unfold_1_2, unfold_1_3, unfold_1_4)
        unfold_2_1 = SMLP(130, 128, True, 'relu')
        unfold_2_2 = SMLP(128, 128, True, 'relu')
        unfold_2_3 = SMLP(128, 64, True, 'relu')
        unfold_2_4 = SMLP(64, 2, False, 'none')
        self.unfold_2 = nn.Sequential(unfold_2_1, unfold_2_2, unfold_2_3, unfold_2_4)
        # NOTE: Do rescale operation or not?
        self.rescale_ebd = rescale_ebd

    def forward(self, pts: torch.Tensor):
        _, N, _ = pts.size()
        # NOTE: Encode points(B,N,3) into a codeword(B,1,d{128}).
        cdw = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, N, 1)
        # NOTE: Apply MLPs to points or embeddings with duplicated codeword.
        ebd_mid = self.unfold_1(torch.cat((cdw_dup, pts), dim=-1))  # (B,N,131{128+3}) -> (B,N,2)
        ebd = self.unfold_2(torch.cat((cdw_dup, ebd_mid), dim=-1))  # (B,N,130{128+2}) -> (B,N,2)
        # NOTE: Regard function 'rescale_pe' as a Sigmoid.
        return rescale_pe(ebd, 0, 1) if self.rescale_ebd else ebd

