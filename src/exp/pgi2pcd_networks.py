from torch import nn

from pointcloud2pgi import FlatteningNet
from utils import *


class FlatNet_PGI2PC_FIT_PCT(nn.Module):
    def __init__(self, N_G, N_C, K):
        super(FlatNet_PGI2PC_FIT_PCT, self).__init__()
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
        # Upsample image four times.
        self.progressive_upsample = nn.Sequential(
            # Channels: 128 -> 96 -> 64(up) -> 32 -> 16(up)
            ResImgSMLP(128, 96),
            ResImgSMLP( 96, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 32x32 Resolution
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResImgSMLP(64, 32),
            ResImgSMLP(32, 16),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # 64x64 Resolution
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.to_points = nn.Sequential(
            ImgSMLP(16, 8, True, "none"),
            ImgSMLP( 8, 3, True, "none"),
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
        guidance_pts = self.from_codeword_to_guidance(v_code)  # [B,256,3]
        # NOTE:
        code_as_img = v_code.permute(0, 2, 1).view(B, 128, self.n_G, self.n_G).contiguous()
        upsampled_img = self.progressive_upsample(code_as_img)  # [B,16,64,64]
        points_img = self.to_points(upsampled_img)  # [B,3,64,64]
        # NOTE: The 3rd argument is useless
        blocks_for_sample, _ = get_pgi_blocks(points_img, self.N_G, self.N_C, 4)  # [B,256,4,4,3]
        complete_pts = sample_points_from_blocks_topk(guidance_pts, blocks_for_sample, 8)
        return guidance_pts, complete_pts


class FlatNet_PGI2PC_FIT_PGI(nn.Module):
    def __init__(self, N_G, N_C, K):
        super(FlatNet_PGI2PC_FIT_PGI, self).__init__()
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

        self.upsample_ftr = nn.Sequential(
            nn.ConvTranspose1d(128, 96, 2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.ConvTranspose1d( 96, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d( 64, 32, 2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )
        self.transform_ftr = SMLP(32, 32, False, 'none')  # [B,Ng,32]
        self.from_ftr_to_points = nn.Sequential(
            SMLP(32, 16, True, 'relu'),
            SMLP(16,  8, True, 'relu'),
            SMLP( 8,  3, True, 'none'),
        )
        self.transform_points = SMLP(3, 3, False, 'none')  # [B*Ng,8,3]
        self.to_pgi = FlatteningNet("../ckpt/comps/g2sd.pth", "../ckpt/comps/s2pf.pth", N_G, N_C, K)
        
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
        # NOTE: FC (Structural Codeword) [B*N_G,96] -> [B*N_G,128] -> [B,256,128] (K=49)
        v_code = self.smlp(torch.cat((v_inner, v_inter, v_outer), dim=-1)).view(B, self.N_G, -1)
        # NOTE: Build coarse point cloud.
        guidance_pts: torch.Tensor = self.from_codeword_to_guidance(v_code)  # [B,256,3]
        # NOTE: Reconstruct PGI
        up_sampled_ftr = self.transform_ftr(self.upsample_ftr(v_code.permute(0, 2, 1)).permute(0, 2, 1))
        points = self.from_ftr_to_points(up_sampled_ftr)
        nb_pts = index_points(points, knn_search(points, guidance_pts, k=8)).view(B * self.N_G, -1, 3)
        final_pts = self.transform_points(nb_pts).view(B, -1, 3)
        pgi_of_reconstructed = self.to_pgi(final_pts)
        return guidance_pts, final_pts, pgi_of_reconstructed


if __name__ == "__main__":
    pass

    N_G = 256
    N_C = 40
    K = 49
    """ Rec """
    data = torch.randn([5, 12544, 3]).cuda()
    # flatnet_rec = FlatNet_PGI2PC_FIT_POINTS(N_G, N_C, K).cuda()
    flatnet_rec = FlatNet_PGI2PC_FIT_PGI(N_G, N_C, K).cuda()
    flatnet_rec(data)
