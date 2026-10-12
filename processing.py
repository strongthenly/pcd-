"""点云后处理：去噪、稠密化、局部结构（高斯椭球）估计、着色辅助。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from pcd_io import PointCloud


@dataclass
class ProcessOptions:
    denoise: bool = True
    denoise_k: int = 16
    denoise_std: float = 2.0
    fill_gaps: bool = True           # 填补扫描线之间的空隙
    max_gap: float = 0.0             # 最大填补距离，0 = 自动估计
    densify_passes: int = 1          # 中点加密轮数，0 = 不加密
    densify_k: int = 6
    max_points: int = 6_000_000      # 稠密化上限，防止内存爆掉
    splat_k: int = 12


def _noop(msg, pct):
    pass


# ------------------------------------------------------------ 去噪
def statistical_outlier_removal(pc: PointCloud, k=16, std_ratio=2.0) -> PointCloud:
    if len(pc) < k + 2:
        return pc
    tree = cKDTree(pc.xyz)
    d, _ = tree.query(pc.xyz, k=k + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return pc.subset(mean_d <= thr)


# ------------------------------------------------------------ 稠密化
def _voxel_dedupe(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """返回每个体素保留的第一个点的索引（保持原点优先）。"""
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    if np.prod(dims.astype(np.float64)) < 2**62:
        lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        _, idx = np.unique(lin, return_index=True)
    else:
        _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def _voxel_keys(xyz, voxel):
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = keys.max(axis=0) + 1
    return (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]


def _voxel_centroids(pc: PointCloud, voxel):
    _, inv, cnt = np.unique(_voxel_keys(pc.xyz, voxel), return_inverse=True, return_counts=True)
    m = len(cnt)

    def mean(a):
        a = a.reshape(len(a), -1).astype(np.float64)
        return np.stack([np.bincount(inv, a[:, c], m) for c in range(a.shape[1])], 1) / cnt[:, None]
    xyz = mean(pc.xyz)
    rgb = None if pc.rgb is None else mean(pc.rgb)
    inten = None if pc.intensity is None else mean(pc.intensity)[:, 0]
    return xyz, rgb, inten


def estimate_gap(xyz: np.ndarray, s: float) -> float:
    """估计扫描线之间的典型间距。

    若表面采样是“线内间距 s、线间距 G”，则边长 V(>G) 的体素内约有 (V/s)(V/G) 个点，
    由此反推 G = V² / (s · 点数)。用按点数加权的中位数，减小边缘体素的影响。
    """
    V = 16 * s
    G = s
    for _ in range(8):
        _, cnt = np.unique(_voxel_keys(xyz, V), return_counts=True)
        cs = np.sort(cnt)
        c = cs[np.searchsorted(np.cumsum(cs), cs.sum() / 2)]
        G = V * V / (s * max(c, 1))
        if G < V / 2:
            break
        V *= 2
    return float(G)


def fill_gaps(pc: PointCloud, max_gap: float = 0.0, max_points=6_000_000, progress=_noop) -> PointCloud:
    """填补扫描线之间（或稀疏区域）的空隙。

    1. 按 max_gap/2.5 体素下采样，得到各向同性的粗结构（激光雷达线内的密集点被合并）；
    2. 在粗结构上找邻居，并在局部切平面上按角度排序，相邻两个邻居与中心点组成三角形；
    3. 只保留三条边都 ≤ max_gap 的三角形（不跨越真实的大空洞 / 不同物体），
       在三角形内按重心坐标均匀撒点，颜色/强度同步插值；
    4. 删掉离原始点太近的新点，原始点全部保留。
    """
    n = len(pc)
    if n < 20:
        return pc
    tree0 = cKDTree(pc.xyz)
    s = float(np.median(tree0.query(pc.xyz[:: max(1, n // 200_000)], k=2, workers=-1)[0][:, 1]))
    if max_gap <= 0:
        g = estimate_gap(pc.xyz, s)
        # 各向同性（随机/均匀）采样时 g≈4~5 倍最近邻距离；明显更大才说明存在“扫描线空隙”
        if g < 8 * s:
            progress(f"点云分布已较均匀（点距≈{s:.3f}），跳过空隙填补", 0)
            return pc
        max_gap = float(min(g * 2.5, 120 * s))
    if max_gap < 3 * s:
        return pc
    v = max_gap / 2.5
    t = max(v / 2, s)  # 新增点的间距
    progress(f"空隙填补：点距≈{s:.3f}，最大填补距离 {max_gap:.3f}，新点间距 {t:.3f}", 0)

    C, Crgb, Cint = _voxel_centroids(pc, v)
    m = len(C)
    k = int(min(12, m - 1))
    if k < 3:
        return pc
    treeC = cKDTree(C)
    tris = []
    chunk = 200_000
    for c0 in range(0, m, chunk):
        c1 = min(m, c0 + chunk)
        tris.append(_triangles_chunk(C, treeC, c0, c1, k, max_gap))
    tri = np.vstack(tris)
    if len(tri) == 0:
        return pc
    if float(m) ** 3 < 9e18:  # 用整数键去重，比 unique(axis=0) 快得多
        key = (tri[:, 0].astype(np.int64) * m + tri[:, 1]) * m + tri[:, 2]
        _, ui = np.unique(key, return_index=True)
        tri = tri[ui]
    else:
        tri = np.unique(tri, axis=0)

    # 在三角形内按网格撒点（分块，边生成边过滤，控制内存）
    P0, P1, P2 = C[tri[:, 0]], C[tri[:, 1]], C[tri[:, 2]]
    emax = np.max(np.stack([np.linalg.norm(P1 - P0, axis=1), np.linalg.norm(P2 - P1, axis=1),
                            np.linalg.norm(P0 - P2, axis=1)], 1), 1)
    del P0, P1, P2
    L = np.clip(np.ceil(emax / t), 1, 12).astype(int)
    budget = max(0, max_points - n)
    new_xyz, new_rgb, new_int = [], [], []
    total = 0
    for lv in np.unique(L):
        if lv < 2:
            continue
        bc = np.array([(a, b, lv - a - b) for a in range(lv + 1) for b in range(lv + 1 - a)], float) / lv
        bc = bc[(bc < 1 - 1e-9).all(1)]  # 不重复生成顶点
        sel_all = np.nonzero(L == lv)[0]
        step = max(1, 2_000_000 // len(bc))
        for b0 in range(0, len(sel_all), step):
            T = tri[sel_all[b0:b0 + step]]
            nx = np.einsum("pb,tbc->tpc", bc, C[T]).reshape(-1, 3)
            dd, _ = tree0.query(nx, k=1, distance_upper_bound=t * 0.6, workers=-1)
            keep = np.isinf(dd)          # 离原始点太近的不要
            new_xyz.append(nx[keep])
            if Crgb is not None:
                new_rgb.append(np.einsum("pb,tbc->tpc", bc, Crgb[T]).reshape(-1, 3)[keep])
            if Cint is not None:
                new_int.append(np.einsum("pb,tb->tp", bc, Cint[T]).ravel()[keep])
            total += int(keep.sum())
            if total > 3 * budget + 1_000_000:
                break
    if not new_xyz:
        return pc
    nx = np.vstack(new_xyz)
    nr = np.vstack(new_rgb) if new_rgb else None
    ni = np.concatenate(new_int) if new_int else None
    if len(nx):
        kk = _voxel_dedupe(nx, t * 0.7)
        if len(kk) > budget:
            kk = np.sort(np.random.default_rng(0).choice(kk, budget, replace=False))
        nx = nx[kk]
        nr = None if nr is None else nr[kk]
        ni = None if ni is None else ni[kk]
    out = PointCloud(
        np.vstack([pc.xyz, nx]),
        None if pc.rgb is None else np.vstack([pc.rgb, np.clip(nr, 0, 255).astype(np.uint8)]),
        None if pc.intensity is None else np.concatenate([pc.intensity, ni.astype(np.float32)]),
        pc.meta)
    progress(f"空隙填补：{n:,} → {len(out):,} 点", 0)
    return out


def _triangles_chunk(C, treeC, c0, c1, k, max_gap):
    """为粗结构点 c0..c1 生成局部三角形（顶点索引，已排序）。"""
    m = c1 - c0
    d, nb = treeC.query(C[c0:c1], k=k + 1, workers=-1)
    d, nb = d[:, 1:], nb[:, 1:]
    valid = d <= max_gap
    nbr = (C[nb] - C[c0:c1, None, :]).astype(np.float32)
    w = valid[..., None]
    cnt = np.maximum(valid.sum(1), 1)[:, None]
    mu = (nbr * w).sum(1) / cnt
    cc = (nbr - mu[:, None]) * w
    _, ev = np.linalg.eigh(cc.transpose(0, 2, 1) @ cc)
    del cc
    e1, e2 = ev[:, :, 2], ev[:, :, 1]
    ang = np.arctan2(np.einsum("nki,ni->nk", nbr, e2), np.einsum("nki,ni->nk", nbr, e1))
    ang = np.where(valid, ang, np.inf)
    order = np.argsort(ang, axis=1)
    nb_s = np.take_along_axis(nb, order, 1)
    ang_s = np.take_along_axis(ang, order, 1)
    nval = valid.sum(1)
    rows = np.arange(m)
    last = np.clip(nval - 1, 0, k - 1)
    with np.errstate(invalid="ignore"):
        dA = np.hstack([ang_s[:, 1:] - ang_s[:, :-1],
                        ang_s[:, :1] + 2 * np.pi - ang_s[rows, last][:, None]])
    A = np.hstack([nb_s[:, :-1], nb_s[rows, last][:, None]]).ravel()
    B = np.hstack([nb_s[:, 1:], nb_s[:, :1]]).ravel()
    ok = np.hstack([np.arange(k - 1)[None, :] + 1 < nval[:, None], (nval >= 3)[:, None]]).ravel()
    dA = np.nan_to_num(dA.ravel(), nan=-1.0)
    ok &= (dA > 1e-3) & (dA < np.radians(150))
    I = np.repeat(rows + c0, k)[ok]
    A, B = A[ok], B[ok]
    ok = np.linalg.norm(C[A] - C[B], axis=1) <= max_gap
    return np.sort(np.stack([I[ok], A[ok], B[ok]], 1), axis=1).astype(np.int64)


def densify(pc: PointCloud, passes=1, k=6, max_points=6_000_000, progress=_noop) -> PointCloud:
    """邻域中点插值稠密化。

    对每个点与其 k 个近邻生成中点（只在“合理距离”内插值，避免跨越空洞/不同物体），
    再按体素去重，使新增点均匀填补点与点之间的空隙。颜色与强度同步插值。
    """
    for p in range(passes):
        n = len(pc)
        if n < k + 2 or n >= max_points:
            break
        tree = cKDTree(pc.xyz)
        d, nb = tree.query(pc.xyz, k=k + 1, workers=-1)
        d, nb = d[:, 1:], nb[:, 1:]
        med = np.median(d[:, 0])
        # 只连接“局部尺度”内的邻居：不超过本点邻域距离中位数的 2 倍，
        # 也不超过全局最近邻间距的 8 倍，避免在不同物体之间的空隙里插点。
        local = np.median(d, axis=1, keepdims=True)
        limit = np.minimum(local * 2.0, med * 8.0)
        i = np.repeat(np.arange(n), k)
        j = nb.ravel()
        ok = (d <= limit).ravel() & (i < j)  # i<j 去掉重复边
        i, j = i[ok], j[ok]
        budget = max_points - n
        if len(i) > budget:
            sel = np.random.default_rng(p).choice(len(i), budget, replace=False)
            i, j = i[sel], j[sel]
        new_xyz = 0.5 * (pc.xyz[i] + pc.xyz[j])
        xyz = np.vstack([pc.xyz, new_xyz])
        rgb = None if pc.rgb is None else np.vstack(
            [pc.rgb, ((pc.rgb[i].astype(np.uint16) + pc.rgb[j]) // 2).astype(np.uint8)])
        inten = None if pc.intensity is None else np.concatenate(
            [pc.intensity, 0.5 * (pc.intensity[i] + pc.intensity[j])])
        keep = _voxel_dedupe(xyz, med * 0.6)
        pc = PointCloud(xyz[keep], None if rgb is None else rgb[keep],
                        None if inten is None else inten[keep], pc.meta)
        progress(f"稠密化 第{p + 1}/{passes}轮: {n:,} → {len(pc):,} 点", 0)
    return pc


# ------------------------------------------------------------ 高斯椭球
def estimate_gaussians(xyz: np.ndarray, k=12, progress=_noop):
    """用局部 PCA 为每个点估计一个 3D 高斯（协方差）。

    返回:
        cov6  (N,6) float32 : 协方差上三角 [xx, xy, xz, yy, yz, zz]
        normals (N,3) float32
        scales (N,3) float32 : 三轴标准差（从大到小）
        rots   (N,3,3) float32 : 列向量为主轴方向
        spacing float : 典型点间距（最近邻距离中位数）
    平坦区域 → 贴合表面的扁圆盘；线状结构 → 细长椭球。
    """
    n = len(xyz)
    k = int(min(k, max(n - 1, 1)))
    tree = cKDTree(xyz)
    cov6 = np.empty((n, 6), np.float32)
    normals = np.empty((n, 3), np.float32)
    scales = np.empty((n, 3), np.float32)
    rots = np.empty((n, 3, 3), np.float32)
    chunk = 400_000
    nn = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, nb = tree.query(xyz[s:e], k=k + 1, workers=-1)
        nbr = xyz[nb]                                  # (m,k+1,3)
        c = (nbr - nbr.mean(axis=1, keepdims=True)).astype(np.float32)
        C = (c.transpose(0, 2, 1) @ c) / (k + 1)
        w, v = np.linalg.eigh(C)                       # 升序
        w, v = w[:, ::-1], v[:, :, ::-1]               # 降序
        # 尺度：主轴取局部点间距，使相邻高斯刚好重叠；其余轴按特征值比例
        spacing = d[:, 1:4].mean(axis=1)
        nn.append(d[:, 1])
        ratio = np.sqrt(np.clip(w / np.maximum(w[:, :1], 1e-20), 0, 1))
        sc = spacing[:, None] * 0.75 * np.clip(ratio, 0.12, 1.0)
        sc[:, 2] = np.minimum(sc[:, 2], spacing * 0.15)  # 法向方向保持很薄
        # 保证右手系
        det = np.linalg.det(v)
        v[det < 0, :, 2] *= -1
        S2 = sc ** 2
        Cg = (v * S2[:, None, :]) @ v.transpose(0, 2, 1)
        cov6[s:e] = np.stack([Cg[:, 0, 0], Cg[:, 0, 1], Cg[:, 0, 2],
                              Cg[:, 1, 1], Cg[:, 1, 2], Cg[:, 2, 2]], 1)
        normals[s:e] = v[:, :, 2]
        scales[s:e] = sc
        rots[s:e] = v
        progress(f"估计高斯椭球 {e:,}/{n:,}", int(100 * e / n))
    return cov6, normals, scales, rots, float(np.median(np.concatenate(nn)))


# ------------------------------------------------------------ 颜色映射
def turbo(t: np.ndarray) -> np.ndarray:
    """Turbo 色图的多项式近似，t∈[0,1] → uint8 RGB。"""
    t = np.clip(t, 0, 1)[:, None]
    kr = [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]
    kg = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]
    kb = [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]
    out = []
    for kk in (kr, kg, kb):
        acc = np.zeros_like(t)
        for c in reversed(kk):
            acc = acc * t + c
        out.append(acc)
    return (np.clip(np.hstack(out), 0, 1) * 255).astype(np.uint8)


def scalar_colors(values: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    return turbo((values - lo) / max(hi - lo, 1e-9))


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) 四元数 wxyz。"""
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4), np.float64)
    w = np.sqrt(np.maximum(0, 1 + tr)) / 2
    x = np.sqrt(np.maximum(0, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) / 2
    x = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q[:, 0], q[:, 1], q[:, 2], q[:, 3] = w, x, y, z
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def export_gaussian_ply(path, xyz, rgb, opacity, scales, rots):
    """导出为标准 3DGS 格式 .ply，可在 SuperSplat 等高斯泼溅查看器中打开。"""
    n = len(xyz)
    C0 = 0.28209479177387814
    f_dc = (rgb.astype(np.float32) / 255.0 - 0.5) / C0
    op = np.clip(opacity, 1e-4, 1 - 1e-4)
    op_logit = np.log(op / (1 - op))
    sc_log = np.log(np.maximum(scales, 1e-7))
    q = rotmat_to_quat_wxyz(rots.astype(np.float64))
    normals = np.zeros((n, 3), np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    data = np.hstack([xyz, normals, f_dc, op_logit[:, None], sc_log, q]).astype(np.float32)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
