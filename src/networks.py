from math import sqrt
from torch import nn

from pointcloud2pgi import FlatteningNet
from utils import *


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors, num_layers):
        super(EdgeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        if self.num_layers == 1:
            self.smlp = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
        if self.num_layers == 2:
            smlp_1 = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            smlp_2 = SMLP(out_channels, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            self.smlp = nn.Sequential(smlp_1, smlp_2)

    def forward(self, pc_ftr):
        num_neighbors = self.num_neighbors
        batch_size, num_points, in_channels = pc_ftr.size()
        knn_indices = knn_search(pc_ftr.detach(), pc_ftr.detach(), num_neighbors)
        nb_ftr = index_points(pc_ftr, knn_indices)
        pc_ftr_rep = pc_ftr.unsqueeze(2).repeat(1, 1, num_neighbors, 1)
        edge_ftr = torch.cat((pc_ftr_rep, nb_ftr-pc_ftr_rep), dim=-1)
        out_ftr = self.smlp(edge_ftr.view(batch_size, num_points*num_neighbors, -1)).view(batch_size, num_points, num_neighbors, -1)
        out_ftr_max_pooled = torch.max(out_ftr, dim=2)[0]
        return out_ftr_max_pooled


class ClsHead(nn.Module):
    def __init__(self, Ci, Nc):
        super(ClsHead, self).__init__()
        self.Ci = Ci
        self.Nc = Nc
        head_dims = [Ci, 512, 256, Nc]
        linear_1 = nn.Linear(head_dims[0], head_dims[1], bias=False)
        bn_1 = nn.BatchNorm1d(head_dims[1])
        nl_1 = nn.LeakyReLU(True, 0.2)
        dp_1 = nn.Dropout(0.5)
        self.fc_1 = nn.Sequential(linear_1, bn_1, nl_1, dp_1)
        linear_2 = nn.Linear(head_dims[1], head_dims[2], bias=False)
        bn_2 = nn.BatchNorm1d(head_dims[2])
        nl_2 = nn.LeakyReLU(True, 0.2)
        dp_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Sequential(linear_2, bn_2, nl_2, dp_2)
        self.fc_3 = nn.Linear(head_dims[2], head_dims[3], bias=False)

    def forward(self, cdw):
        Ci, Nc = self.Ci, self.Nc
        B, D, device = cdw.size(0), cdw.size(1), cdw.device
        logits = self.fc_3(self.fc_2(self.fc_1(cdw)))
        return logits


class FlatNetCls(nn.Module):
    def __init__(self, N_G, N_C, K, num_classes):
        super(FlatNetCls, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        self.n_G = int(N_G ** 0.5)
        self.k = int(K ** 0.5)
        self.M = N_G * K
        self.m = self.n_G * self.k
        # [4,7,7] (K=49): center, inner, inter, outer
        self.csm = get_concentric_square_masks(self.k)
        # [7,7] (K=49): inner'(center U inner), inter, outer
        self.inner_mask, self.inter_mask, self.outer_mask = merge_concentric_square_masks(self.csm)
        self.num_inner = int(self.inner_mask.sum().item())  #  9 (K=49)
        self.num_inter = int(self.inter_mask.sum().item())  # 16 (K=49)
        self.num_outer = int(self.outer_mask.sum().item())  # 24 (K=49)
        # Increase channel size from 3 to 32
        self.lift = SMLP(3, 32, True, 'leakyrelu', 0.20)
        # Codeword
        self.smlp = FC(96, 128, True, 'leakyrelu', 0.20)
        self.edge_conv_1 = EdgeConv(128, 128, 16, 1)
        self.edge_conv_2 = EdgeConv(128, 128, 16, 1)
        self.edge_conv_3 = EdgeConv(128, 256, 16, 1)
        self.fuse = SMLP(512, 1024, True, 'leakyrelu', 0.20)
        self.head = ClsHead(2048, num_classes)

    def forward(self, pgi):
        B = pgi.size(0)
        device = pgi.device
        # NOTE: Convert PGI from list [B,M,3] to image [B,3,m,m].
        I = pgi.permute(0, 2, 1).contiguous().view(B, 3, self.m, self.m)
        # NOTE: Partition
        blocks, _ = get_pgi_blocks(I, self.N_G, self.N_C, self.k)  # [B,256,7,7,3] (K=49)
        blk_pts_inner = square_partition(blocks, self.inner_mask.to(device))  # [B,256, 9,3] (K=49)
        blk_pts_inter = square_partition(blocks, self.inter_mask.to(device))  # [B,256,16,3] (K=49)
        blk_pts_outer = square_partition(blocks, self.outer_mask.to(device))  # [B,256,24,3] (K=49)
        # NOTE: Square-wise Embedding [B*N_G,-1,3] -(channel-wise maxpool)-> [B*N_G,32]
        v_inner = self.lift(blk_pts_inner.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_inter = self.lift(blk_pts_inter.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_outer = self.lift(blk_pts_outer.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        # NOTE: FC (Structural Codeword) [B,256,128] (K=49)
        ftr_0 = self.smlp(torch.cat((v_inner, v_inter, v_outer), dim=-1)).view(B, self.N_G, -1)

        # NOTE: Edge-Style Convolutions...(DGCNN)
        ftr_1 = self.edge_conv_1(ftr_0)  # [5,256,128]
        ftr_2 = self.edge_conv_2(ftr_1)  # [b,256,128]
        ftr_3 = self.edge_conv_3(ftr_2)  # [5,256,256]
        ftr = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3), dim=-1))
        cdw = torch.cat((ftr.max(dim=1)[0], ftr.mean(dim=1)), dim=-1)
        logits = self.head(cdw)
        return logits


class FlatNetRec(nn.Module):
    def __init__(self, N_G, N_C, K):
        super(FlatNetRec, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        self.n_G = int(N_G ** 0.5)
        self.k = int(K ** 0.5)
        self.M = N_G * K
        self.m = self.n_G * self.k
        # [4,7,7] (K=49): center, inner, inter, outer
        self.csm = get_concentric_square_masks(self.k)
        # [7,7] (K=49): inner'(center U inner), inter, outer
        self.inner_mask, self.inter_mask, self.outer_mask = merge_concentric_square_masks(self.csm)
        self.num_inner = int(self.inner_mask.sum().item())  #  9 (K=49)
        self.num_inter = int(self.inter_mask.sum().item())  # 16 (K=49)
        self.num_outer = int(self.outer_mask.sum().item())  # 24 (K=49)
        # Increase channel size from 3 to 32
        self.lift = SMLP(3, 32, True, 'leakyrelu', 0.20)
        # Codeword
        self.smlp = FC(96, 128, True, 'leakyrelu', 0.20)
        # To Coarse Point Cloud   [B,N,C] -> [B,N,3]
        self.from_codeword_to_guidance = nn.Sequential(
            SMLP(128, 64, True, 'relu'),
            SMLP( 64, 32, True, 'relu'),
            SMLP( 32, 16, True, 'relu'),
            SMLP( 16,  3, True, 'none'),
        )


    def forward(self, pgi):
        B, device = pgi.size(0), pgi.device
        # NOTE: Convert PGI from list [B,M,3] to image [B,3,m,m].
        I = pgi.permute(0, 2, 1).contiguous().view(B, 3, self.m, self.m)
        # NOTE: Partition
        blocks, _ = get_pgi_blocks(I, self.N_G, self.N_C, self.k)  # [B,256,7,7,3] (K=49)
        blk_pts_inner = square_partition(blocks, self.inner_mask.to(device))  # [B,256, 9,3] (K=49)
        blk_pts_inter = square_partition(blocks, self.inter_mask.to(device))  # [B,256,16,3] (K=49)
        blk_pts_outer = square_partition(blocks, self.outer_mask.to(device))  # [B,256,24,3] (K=49)
        # NOTE: Square-wise Embedding [B*N_G,-1,3] -(channel-wise maxpool)-> [B*N_G,32]
        v_inner = self.lift(blk_pts_inner.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_inter = self.lift(blk_pts_inter.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_outer = self.lift(blk_pts_outer.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        # NOTE: FC (Structural Codeword) [B,256,128] (K=49)
        v_code = self.smlp(torch.cat((v_inner, v_inter, v_outer), dim=-1)).view(B, self.N_G, -1)
        # NOTE: Build coarse point cloud.
        guidance_pts = self.from_codeword_to_guidance(v_code)  # [B,256,3] (Sparse)
        return guidance_pts


if __name__ == "__main__":
    pass

    N_G = 256
    N_C = 40
    K = 49
    """ Cls """
    # data = torch.randn([5, 12544, 3]).cuda()
    # flatnet_cls = FlatNetCls(N_G, N_C, K, 10).cuda()
    # flatnet_cls(data)
