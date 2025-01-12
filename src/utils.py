import sys
sys.path.append("..")

import numpy as np
import torch
import itertools
import torch.nn.functional as F

from PIL import Image
from torch import nn
from scipy.spatial.transform import Rotation as scipy_R

from libs.CD.chamferdist.chamfer import knn_points as knn_gpu
from libs.EMD.emd import earth_mover_distance_unwrapped


def range_function(x, t_min, t_max):
    y = F.relu(-x+t_min) + F.relu(x-t_max)
    return y


def square_dist_self(P):
    B, N, C = P.shape
    device = P.device
    D = -2 * torch.matmul(P, P.permute(0, 2, 1))
    D += torch.sum(P**2, -1).view(B, N, 1)
    D += torch.sum(P**2, -1).view(B, 1, N)
    Z = (1 - torch.diag(torch.ones(N))).float().to(device)
    D *= Z
    return torch.abs(D)


def unfm_loss(x, t_min, t_max):
    B, N, C = x.size()
    device = x.device
    t_min_squared = t_min ** 2
    t_max_squared = t_max ** 2
    diag_mask = torch.diag(torch.ones(N) * 1e3).float().unsqueeze(0).repeat(B, 1, 1).to(device)
    pwd_squared = square_dist_self(x)
    nearest_neighbor_distances_squared = (pwd_squared + diag_mask).sort(dim=-1)[0][:, :, 0]
    errors_squared = range_function(nearest_neighbor_distances_squared, t_min_squared, t_max_squared)
    return errors_squared.mean(dim=-1).mean()


def sample_points_from_blocks_topk(query, blocks, target_num_points):
    # query: [B,N,3], blocks: [B,N,r,r,3], output: [B,256,8,3] -> [B,256*8,3]
    B, N = query.size(0), query.size(1)
    query = query.unsqueeze(2).unsqueeze(3)  # [B,N,1,1,3]
    dist = torch.sum(((blocks - query) ** 2), dim=-1)  # [B,N,H,W]
    # [B,N,target_num_points]
    _, idx = torch.topk(dist.view(B, N, -1), k=target_num_points, dim=-1, largest=False)
    # [B,N,target_num_points,3]
    sampled_pts = torch.gather(blocks.view(B, N, -1, 3), dim=2, index=idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    return sampled_pts.view(B, N * target_num_points, -1)


def get_pgi_blocks(I, N_G, N_C, k):
    assert I.size(2) == I.size(3)
    K = k * k
    B, _, _, m = I.size()
    assert np.mod(m, k) == 0
    blocks = F.unfold(I, kernel_size=k, stride=k)
    blocks = blocks.view(B, 3, k, k, N_G)
    blocks = blocks.permute(0, 4, 2, 3, 1).contiguous()
    blocks_n = normalize_anchor_patches(blocks.view(B, N_G, K, 3)).view(B, N_G, k, k, 3)
    return blocks, blocks_n


def get_pgi_ftr_blocks(I, N_G, N_C, k):
    assert I.size(2) == I.size(3)
    K = k * k
    B, C, _, m = I.size()
    assert np.mod(m, k) == 0
    blocks = F.unfold(I, kernel_size=k, stride=k)
    blocks = blocks.view(B, C, k, k, N_G)
    blocks = blocks.permute(0, 4, 2, 3, 1).contiguous()
    return blocks


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    xn = (x - x_min) / (x_max - x_min)
    return xn


def show_image(img):
    # input type: torch.tensor or numpy.ndarray
    # input size: 3xHxW or 1xHxW or HxW
    assert img.ndim in [2, 3]
    if img.ndim == 3:
        assert img.shape[0] in [1, 3]
    img = (min_max_normalization(np.asarray(img)) * 255.0).astype(np.uint8)
    if img.ndim==2 or img.shape[0]==1:
        img = img.squeeze() # HxW
        img_pil = Image.fromarray(img).convert('L')
    else:
        img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
    return img_pil


def visualize_pgi(pgi, scaling_factor=1):
    # pgi: [batch_size, num_pixels, 3]
    assert pgi.ndim == 3
    assert pgi.size(2) == 3
    bs, num_pixels = pgi.size(0), pgi.size(1)
    is_square_number(num_pixels)
    img_res = int(np.sqrt(num_pixels))
    pgi = np.asarray(pgi.cpu().permute(0, 2, 1).contiguous().view(bs, -1, img_res, img_res)) # (bs, 3, img_res, img_res)
    images_list = []
    for bid in range(bs):
        img = show_image(pgi[bid])
        resized_img_res = int(np.around(img_res * scaling_factor))
        img = img.resize((resized_img_res, resized_img_res), Image.BILINEAR)
        images_list.append(img)
    return images_list


def repulsion_function(x, t_min):
    y = F.relu(-x + t_min)
    return y


def repulsion_loss(pe, min_nnd):
    # pe: [B, N, 2], planar embeddings
    # min_nnd: a pre-defined threshold that specifies the minimum L2 distance between nearest neighbors
    assert pe.size(2) == 2
    B = pe.size(0)
    N = pe.size(1)
    nn_idx = knn_search(pe, pe, 2)[:, :, 1] # (B, N)
    pe_nn = index_points(pe, nn_idx) # [B, N, 2]
    nnd_squared = torch.sum((pe-pe_nn)**2, dim=-1) # [B, N], nearest neighbor distances (squared)
    min_nnd_squared = min_nnd ** 2
    err = repulsion_function(nnd_squared, min_nnd_squared) # [B, N]
    rep_loss = err.mean()
    return rep_loss


def random_axis_rotation(pc, axis, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert axis in ['x', 'y', 'z']
    # generate a random rotation matrix
    angle = np.random.uniform() * 2 * np.pi
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        return pc_rotated, angle


def random_rotation(pc, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    rot_mat = scipy_R.random().as_matrix().astype(np.float32) # (3, 3)
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        rot_ang = scipy_R.from_matrix(np.transpose(rot_mat)).as_euler('xyz', degrees=True).astype(np.float32) # (3,)
        for aid in range(3):
            if rot_ang[aid] < 0:
                rot_ang[aid] = 360.0 + rot_ang[aid]
        return pc_rotated, rot_ang


def parse_list_file(list_file_path):
    list_file = [line.strip() for line in open(list_file_path, 'r')]
    return list_file


def chamfer_distance_cuda(pts_s, pts_t, cpt_mode='max', return_detail=False):
    # pts_s: [B, Ns, C], source point cloud
    # pts_t: [B, Nt, C], target point cloud
    Bs, Ns, Cs, device_s = pts_s.size(0), pts_s.size(1), pts_s.size(2), pts_s.device
    Bt, Nt, Ct, device_t = pts_t.size(0), pts_t.size(1), pts_t.size(2), pts_t.device
    assert Bs == Bt
    assert Cs == Ct
    assert device_s == device_t
    assert device_s.type == 'cuda' and device_t.type == 'cuda'
    assert cpt_mode in ['max', 'avg']
    lengths_s = torch.ones(Bs, dtype=torch.long, device=device_s) * Ns
    lengths_t = torch.ones(Bt, dtype=torch.long, device=device_t) * Nt
    source_nn = knn_gpu(pts_s, pts_t, lengths_s, lengths_t, 1)
    target_nn = knn_gpu(pts_t, pts_s, lengths_t, lengths_s, 1)
    source_dist, source_idx = source_nn.dists.squeeze(-1), source_nn.idx.squeeze(-1) # [B, Ns]
    target_dist, target_idx = target_nn.dists.squeeze(-1), target_nn.idx.squeeze(-1) # [B, Nt]
    batch_dist = torch.cat((source_dist.mean(dim=-1, keepdim=True), target_dist.mean(dim=-1, keepdim=True)), dim=-1) # [B, 2]
    if cpt_mode == 'max':
        cd = batch_dist.max(dim=-1)[0].mean()
    if cpt_mode == 'avg':
        cd = batch_dist.mean(dim=-1).mean()
    if not return_detail:
        return cd
    else:
        return cd, source_dist, source_idx, target_dist, target_idx


def earth_mover_distance_cuda(pts_1, pts_2):
    # pts_1: [B, N1, C=1,2,3]
    # pts_2: [B, N2, C=1,2,3]
    assert pts_1.size(0) == pts_2.size(0)
    assert pts_1.size(2) == pts_2.size(2)
    assert pts_1.device == pts_2.device
    B, N1, C, device = pts_1.size(0), pts_1.size(1), pts_1.size(2), pts_1.device
    B, N2, C, device = pts_2.size(0), pts_2.size(1), pts_2.size(2), pts_2.device
    assert device.type == 'cuda'
    assert C in [1, 2, 3]
    if C < 3:
        pts_1 = torch.cat((pts_1, torch.zeros(B, N1, 3-C).to(device)), dim=-1) # [B, N1, 3]
        pts_2 = torch.cat((pts_2, torch.zeros(B, N2, 3-C).to(device)), dim=-1) # [B, N2, 3]  
    # double direction
    dist_1 = earth_mover_distance_unwrapped(pts_1, pts_2, transpose=False) / N1 # [B]
    dist_2 = earth_mover_distance_unwrapped(pts_2, pts_1, transpose=False) / N2 # [B]
    emd = ((dist_1 + dist_2) / 2).mean()
    # single direction
    # dist = earth_mover_distance_unwrapped(pts_1, pts_2, transpose=False)
    # emd = (dist / N1).mean()
    return emd


def compute_smooth_cross_entropy(pred, label, eps=0.2):
    #  Cross Entropy Loss with Label Smoothing
    # label = label.contiguous().view(-1)
    # pred: [batch_size, num_classes]
    # label: [batch_size]
    num_classes = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(pred, dim=1)
    sce_loss = -(one_hot * log_prb).sum(dim=1).mean()
    return sce_loss


def mask_selection_2d(inputs, binary_mask):
    assert inputs.size(2)==binary_mask.size(0) and inputs.size(3)==binary_mask.size(1)
    B, C, H, W = inputs.size()
    binary_mask_vec = binary_mask.unsqueeze(0).repeat(B*C, 1, 1).view(B*C, H*W)
    inputs_selected = torch.masked_select(inputs.view(B*C, H*W), binary_mask_vec).view(B, C, -1)
    return inputs_selected


def square_partition(gi_blocks, binary_mask):
    assert gi_blocks.size(2) == gi_blocks.size(3)
    B, N_G, k, _, C = gi_blocks.size()
    inputs = gi_blocks.view(B*N_G, k, k, C).permute(0, 3, 1, 2).contiguous()
    inputs_selected = mask_selection_2d(inputs, binary_mask.bool()).view(B, N_G, C, -1)
    inputs_selected = inputs_selected.permute(0, 1, 3, 2).contiguous()
    return inputs_selected


def get_pgi_blocks(I, N_G, N_C, k):
    assert I.size(2) == I.size(3)
    K = k * k
    B, _, _, m = I.size()
    assert np.mod(m, k) == 0
    blocks = F.unfold(I, kernel_size=k, stride=k)
    blocks = blocks.view(B, 3, k, k, N_G)
    blocks = blocks.permute(0, 4, 2, 3, 1).contiguous()
    blocks_n = normalize_anchor_patches(blocks.view(B, N_G, K, 3)).view(B, N_G, k, k, 3)
    return blocks, blocks_n


def merge_concentric_square_masks(csm):
    assert csm.size(1) == csm.size(2)
    num_squares, k, _ = csm.size()
    device = csm.device
    assert k >= 5
    a = int(np.floor(num_squares / 3))
    b = np.mod(num_squares, 3)
    N1 = a + int(np.ceil(b / 2))
    N2 = a + int(np.floor(b / 2))
    N3 = a
    inner_mask = csm[:N1].sum(dim=0)
    inter_mask = csm[N1:N1+N2].sum(dim=0)
    outer_mask = csm[N1+N2:].sum(dim=0)
    return inner_mask.to(device), inter_mask.to(device), outer_mask.to(device)


def get_concentric_square_masks(k):
    assert k >= 5
    num_squares = int(np.ceil(k / 2))
    flag_map = np.ones((num_squares, k, k))
    for sid in range(1, num_squares):
        for i in range(sid, k-sid):
            for j in range(sid, k-sid):
                flag_map[sid, i, j] = 0
    flag_map = torch.Tensor(flag_map.sum(axis=0) - 1)
    csm = []
    for sid in range(num_squares):
        csm.append((flag_map == sid).float().unsqueeze(0))
    csm = torch.cat(csm, dim=0)
    return csm


def assemble_separate_patch_parameterizations(sep_pat_para):
    # sep_pat_para: [B, M, 3, r, r]
    assert sep_pat_para.size(2) == 3
    assert sep_pat_para.size(3) == sep_pat_para.size(4)
    B = sep_pat_para.size(0)
    M = sep_pat_para.size(1)
    r = sep_pat_para.size(3)
    device = sep_pat_para.device
    is_square_number(M)
    m = int(np.sqrt(M))
    s = m * r # resolution of raw pgi
    raw_pgi = torch.empty(B, 3, s, s).to(device) # [B, 3, m, m]
    for i in range(m):
        for j in range(m):
            raw_pgi[:, :, i*r:(i+1)*r, j*r:(j+1)*r] = sep_pat_para[:, i*m+j, ...]
    return raw_pgi


def grid_resample(pts, pe, num_rsp):
    assert pts.ndim==3 and pe.ndim==3
    assert pts.size(0)==pe.size(0) and pts.size(1)==pe.size(1)
    assert pts.size(2)==3 and pe.size(2)==2
    is_square_number(num_rsp)
    lat_res = int(np.sqrt(num_rsp))
    batch_size, num_points, device = pe.size(0), pe.size(1), pe.device
    grid_points, _ = build_lattice(lat_res, lat_res)
    grid_points = torch.tensor(grid_points).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    nn_idx = knn_search(pe.detach(), grid_points.detach(), 1).squeeze(-1)
    pts_rsp = index_points(pts, nn_idx)
    return pts_rsp


def seperately_grid_resample_patches(pat_pts, pe, R):
    # pat_pts: [B, M, K, 3], local patch points
    # pe: [B, M, K, 2], planar embeddings
    # R: number of resampled pixels (points)
    is_square_number(R)
    r = int(np.sqrt(R))
    assert pat_pts.size(0) == pe.size(0)
    assert pat_pts.size(1) == pe.size(1)
    assert pat_pts.size(2) == pe.size(2)
    assert pat_pts.size(3)==3 and pe.size(3)==2
    B = pat_pts.size(0)
    M = pat_pts.size(1)
    K = pat_pts.size(2)
    sep_pat_para = grid_resample(pat_pts.view(B*M, K, 3), pe.view(B*M, K, 2), R) # [B*M, R, 3]
    sep_pat_para = sep_pat_para.permute(0, 2, 1).contiguous().view(B*M, 3, r, r).view(B, M, 3, r, r) # [B, M, 3, r, r]
    return sep_pat_para


def normalize_anchor_patches(anc_pat):
    # anc_pat: [B, M, K, C], local patches around anchors
    B = anc_pat.size(0) # batch size
    M = anc_pat.size(1) # number of anchors in the whole point cloud
    K = anc_pat.size(2) # number of points in a local patch
    C = anc_pat.size(3) # number of channels, coordinates (and other attributes, e.g., normals, colors)
    # by default, point coordinates are stored in the first three channels
    assert anc_pat.ndim==4 and C>=3
    coor = anc_pat[:, :, :, 0:3] # [B, M, K, 3], coordinates
    if C > 3:
        atr = anc_pat[:, :, :, 3:].view(B, M, K, -1) # [B, M, K, C-3], attributes
    coor_nrm = coor.view(B*M, K, 3) # [B*M, K, 3]
    centroids = torch.mean(coor_nrm, dim=1, keepdim=True) # [B*M, 1, 3]
    coor_nrm = coor_nrm - centroids # [B*M, K, 3]
    distances = torch.sqrt(torch.abs(torch.sum(coor_nrm**2, dim=-1, keepdim=True))) # [B*M, K, 1]
    max_distances = torch.max(distances, dim=1, keepdim=True)[0] # [B*M, 1, 1]
    coor_nrm = coor_nrm / max_distances # [B*M, K, 3]
    coor_nrm = coor_nrm.view(B, M, K, 3) # [B, M, K, 3]
    if C == 3:
        anc_pat_nrm = coor_nrm # [B, M, K, C=3]
    else: # i.e., C > 3
        anc_pat_nrm = torch.cat((coor_nrm, atr), dim=-1) # [B, M, K, C>3]
    return anc_pat_nrm


def fps(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def is_square_number(n):
    sr = int(np.sqrt(n))
    assert n == (sr ** 2)


def rescale_pe(pe, range_min, range_max):
    # pe: [batch_size, num_points, 2], 2D embeddings
    assert pe.ndim==3 and pe.size(2)==2
    assert range_min < range_max
    values_min = torch.min(pe, dim=1, keepdim=True)[0] # [batch_size, 1, 2]
    values_max = torch.max(pe, dim=1, keepdim=True)[0] # [batch_size, 1, 2]
    # [batch_size, num_points, 2], in the range of [0, 1]
    pe_rescaled = (pe - values_min) / (values_max - values_min)
    # [batch_size, num_points, 2], in the range of [range_min, range_max]
    pe_rescaled = pe_rescaled * (range_max - range_min) + range_min 
    return pe_rescaled


def build_lattice(H, W):
    N = H * W # number of grids
    # generate grid points within the range of (0, 1)
    margin = 1e-4
    h_p = np.linspace(0+margin, 1-margin, H, dtype=np.float32)
    w_p = np.linspace(0+margin, 1-margin, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))) # (N, 2)
    # generate grid indices
    h_i = np.linspace(0, H-1, H, dtype=np.int64)
    w_i = np.linspace(0, W-1, W, dtype=np.int64)
    grid_indices = np.array(list(itertools.product(h_i, w_i))) # (N, 2)
    return grid_points, grid_indices


def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected


def knn_on_gpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx


def knn_search(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    knn_idx = knn_on_gpu(source_pts, query_pts, k)
    return knn_idx


def bounding_box_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def random_translation(pc, max_offset, return_offset=False):
    # pc: (num_points, num_channels)
    # pc_translated: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert max_offset > 0
    offset = np.random.uniform(low=-max_offset, high=max_offset, size=[3]).astype('float32')
    pc_translated = pc.copy()
    pc_translated[:, 0] += offset[0]
    pc_translated[:, 1] += offset[1]
    pc_translated[:, 2] += offset[2]
    if not return_offset:
        return pc_translated
    else:
        return pc_translated, offset


def random_anisotropic_scaling(pc, min_s_ratio, max_s_ratio, return_aniso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_aniso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    aniso_scaling_ratio = (np.random.random(3) * (max_s_ratio - min_s_ratio) + min_s_ratio).astype('float32')
    pc_aniso_scaled = pc.copy()
    pc_aniso_scaled[:, 0] *= aniso_scaling_ratio[0]
    pc_aniso_scaled[:, 1] *= aniso_scaling_ratio[1]
    pc_aniso_scaled[:, 2] *= aniso_scaling_ratio[2]
    if not return_aniso_scaling_ratio:
        return pc_aniso_scaled
    else:
        return pc_aniso_scaled, aniso_scaling_ratio


class SMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(SMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, N, ic]
        # y: [B, N, oc]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [B, ic, N, 1]
        y = self.conv(x) # [B, oc, N, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() # [B, N, oc]
        return y


class ResSMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smlp_1 = SMLP(in_channels, in_channels, True, 'none')
        self.smlp_2 = SMLP(in_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = SMLP(in_channels, out_channels, True, 'none')
        self.nl = nn.ReLU(inplace=True)

    def forward(self, in_ftr):
        # in_ftr: [B, N, in_channels]
        out_ftr = self.smlp_2(self.nl(self.smlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr # [B, N, out_channels]


class CU(nn.Module):
    def __init__(self, ic, oc, ks, is_bn, nl, slope=None, pad='zeros'):
        super(CU, self).__init__()
        assert np.mod(ks + 1, 2) == 0
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        assert pad in ['zeros', 'reflect', 'replicate', 'circular']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=1, 
                    padding=(ks-1)//2, dilation=1, groups=1, bias=False, padding_mode=pad)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, ic, H, W]
        # y: [B, oc, H, W]
        y = self.conv(x) # [B, oc, H, W]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, ic]
        # y: [B, oc]
        y = self.linear(x) # [B, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class ImgSMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(ImgSMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, N, ic]
        # y: [B, N, oc]
        y = self.conv(x) # [B, oc, N, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        return y


class ResImgSMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.out_channels = out_channels
        self.in_channels = in_channels
        super(ResImgSMLP, self).__init__()
        self.smlp_1 = ImgSMLP(in_channels, in_channels, True, 'none')
        self.smlp_2 = ImgSMLP(in_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = ImgSMLP(in_channels, out_channels, True, 'none')
        self.nl = nn.ReLU(inplace=True)

    def forward(self, in_ftr):
        # in_ftr: [B, N, in_channels]
        out_ftr = self.smlp_2(self.nl(self.smlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr # [B, N, out_channels]
