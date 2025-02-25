import numpy as np
from scipy.spatial import cKDTree

def get_bilinear_neighbors(i, j, kd_tree, valid_coords, depth):
    r = 1
    max_r = 50
    while r < max_r:
        indices = kd_tree.query_ball_point([i, j], r)
        if len(indices) == 0:
            r *= 2
            continue
        tl, tr, bl, br = [], [], [], []
        for idx in indices:
            vi, vj = valid_coords[idx]
            if vi <= i and vj <= j:
                tl.append((vi, vj))
            if vi <= i and vj >= j:
                tr.append((vi, vj))
            if vi >= i and vj <= j:
                bl.append((vi, vj))
            if vi >= i and vj >= j:
                br.append((vi, vj))
        if tl and tr and bl and br:
            tl_candidate = max(tl, key=lambda p: (p[0], p[1]))
            tr_candidate = max(tr, key=lambda p: (p[0], -p[1]))
            bl_candidate = min(bl, key=lambda p: (p[0], -p[1]))
            br_candidate = min(br, key=lambda p: (p[0], p[1]))
            return tl_candidate, tr_candidate, bl_candidate, br_candidate
        r *= 2
    return None

def bilinear_interpolation_kdtree(depth):
    depth_filled = depth.copy()
    H, W = depth.shape
    valid_mask = (depth > 0)
    valid_coords = np.argwhere(valid_mask).astype(np.int32)
    if valid_coords.shape[0] == 0:
        return depth_filled
    kd_tree = cKDTree(valid_coords)
    invalid_coords = np.argwhere(~valid_mask)
    for (i, j) in invalid_coords:
        neighbors = get_bilinear_neighbors(i, j, kd_tree, valid_coords, depth)
        if neighbors is None:
            dist, idx = kd_tree.query([i, j])
            ni, nj = valid_coords[idx]
            depth_filled[i, j] = depth[ni, nj]
        else:
            tl, tr, bl, br = neighbors
            i_top = max(tl[0], tr[0])
            i_bottom = min(bl[0], br[0])
            j_left = max(tl[1], bl[1])
            j_right = min(tr[1], br[1])
            if i_top > i or i_bottom < i or j_left > j or j_right < j or (i_bottom - i_top) == 0 or (j_right - j_left) == 0:
                dist, idx = kd_tree.query([i, j])
                ni, nj = valid_coords[idx]
                depth_filled[i, j] = depth[ni, nj]
            else:
                f00 = depth[i_top, j_left]
                f10 = depth[i_top, j_right]
                f01 = depth[i_bottom, j_left]
                f11 = depth[i_bottom, j_right]
                t = (j - j_left) / (j_right - j_left)
                u = (i - i_top) / (i_bottom - i_top)
                interp_val = (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11
                depth_filled[i, j] = interp_val
    return depth_filled

def pixel_to_3d(u, v, depth_val, fx, fy, cx, cy):
    X = (u - cx) * depth_val / fx
    Y = (v - cy) * depth_val / fy
    Z = depth_val
    return X, Y, Z
