"""利用相机图像给点云着色（多视角加权融合 + 遮挡判断）。

支持两种相机数据：
  1. COLMAP 重建结果（cameras.txt/bin + images.txt/bin + 图像目录）
     —— 3D 高斯泼溅训练常用的格式。
  2. 自定义 JSON（适合激光雷达 + 相机标定数据），格式见 README.md。

相机坐标系约定：OpenCV（x 向右、y 向下、z 朝前）。
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter


@dataclass
class CameraView:
    name: str
    image_path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "PINHOLE"            # PINHOLE / OPENCV / RADIAL / FISHEYE
    dist: list = field(default_factory=list)
    R: np.ndarray = None              # world → camera 旋转
    t: np.ndarray = None              # world → camera 平移

    @property
    def center(self):
        return -self.R.T @ self.t


# ================================================================ COLMAP
_COLMAP_MODELS = {  # id: (name, num_params)
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def _intrinsics_from_colmap(model, p):
    """→ fx, fy, cx, cy, model_kind, dist"""
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2], "PINHOLE", []
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3], "PINHOLE", []
    if model == "SIMPLE_RADIAL":
        return p[0], p[0], p[1], p[2], "OPENCV", [p[3], 0, 0, 0]
    if model == "RADIAL":
        return p[0], p[0], p[1], p[2], "OPENCV", [p[3], p[4], 0, 0]
    if model in ("OPENCV", "FULL_OPENCV"):
        return p[0], p[1], p[2], p[3], "OPENCV", list(p[4:8]) + ([p[8]] if len(p) > 8 else [])
    if model == "OPENCV_FISHEYE":
        return p[0], p[1], p[2], p[3], "FISHEYE", list(p[4:8])
    if model in ("SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2], "FISHEYE", list(p[3:5]) + [0, 0]
    if model == "THIN_PRISM_FISHEYE":  # 近似：忽略薄棱镜项
        return p[0], p[1], p[2], p[3], "FISHEYE", [p[4], p[5], p[8], p[9]]
    # FOV 等其它模型：退化为针孔（会有一定畸变误差）
    return p[0], p[1], p[2], p[3], "PINHOLE", []


def _qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def _read_colmap_cameras(model_dir):
    cams = {}
    bin_p, txt_p = os.path.join(model_dir, "cameras.bin"), os.path.join(model_dir, "cameras.txt")
    if os.path.exists(bin_p):
        with open(bin_p, "rb") as f:
            (num,) = struct.unpack("<Q", f.read(8))
            for _ in range(num):
                cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
                name, npar = _COLMAP_MODELS[mid]
                params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
                cams[cid] = (name, int(w), int(h), list(params))
    elif os.path.exists(txt_p):
        with open(txt_p, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                tok = line.split()
                cams[int(tok[0])] = (tok[1], int(tok[2]), int(tok[3]), [float(v) for v in tok[4:]])
    else:
        raise FileNotFoundError(f"{model_dir} 中没有 cameras.bin / cameras.txt")
    return cams


def _read_colmap_images(model_dir):
    imgs = []
    bin_p, txt_p = os.path.join(model_dir, "images.bin"), os.path.join(model_dir, "images.txt")
    if os.path.exists(bin_p):
        with open(bin_p, "rb") as f:
            (num,) = struct.unpack("<Q", f.read(8))
            for _ in range(num):
                vals = struct.unpack("<idddddddi", f.read(64))
                name = b""
                while (c := f.read(1)) != b"\x00":
                    name += c
                (n2d,) = struct.unpack("<Q", f.read(8))
                f.seek(24 * n2d, 1)
                imgs.append((np.array(vals[1:5]), np.array(vals[5:8]), vals[8],
                             name.decode("utf-8")))
    elif os.path.exists(txt_p):
        with open(txt_p, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            tok = lines[i].split()
            imgs.append((np.array([float(v) for v in tok[1:5]]),
                         np.array([float(v) for v in tok[5:8]]),
                         int(tok[8]), " ".join(tok[9:])))
            i += 2  # 第二行是 2D 特征点（可能为空行）
    else:
        raise FileNotFoundError(f"{model_dir} 中没有 images.bin / images.txt")
    return imgs


def find_colmap_dirs(folder):
    """在用户选择的文件夹里自动找 COLMAP 模型目录和图像目录。"""
    cands = [folder, os.path.join(folder, "sparse", "0"), os.path.join(folder, "sparse"),
             os.path.join(folder, "colmap", "sparse", "0"), os.path.join(folder, "distorted", "sparse", "0")]
    model_dir = None
    for c in cands:
        if any(os.path.exists(os.path.join(c, f"cameras.{e}")) for e in ("bin", "txt")):
            model_dir = c
            break
    if model_dir is None:
        return None, None
    img_cands = [os.path.join(folder, "images"), os.path.join(model_dir, "images"),
                 os.path.join(model_dir, "..", "images"), os.path.join(model_dir, "..", "..", "images"),
                 folder]
    for c in img_cands:
        if os.path.isdir(c) and any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
                                    for f in os.listdir(c)):
            return model_dir, os.path.normpath(c)
    return model_dir, None


def load_colmap(model_dir, images_dir) -> list[CameraView]:
    cams = _read_colmap_cameras(model_dir)
    views = []
    for q, t, cid, name in _read_colmap_images(model_dir):
        model, w, h, p = cams[cid]
        fx, fy, cx, cy, kind, dist = _intrinsics_from_colmap(model, p)
        views.append(CameraView(name, os.path.join(images_dir, name), w, h, fx, fy, cx, cy,
                                kind, dist, _qvec_to_R(q / np.linalg.norm(q)), t.astype(np.float64)))
    return views


# ================================================================ JSON
def _intr(d, fallback=None):
    d = {**(fallback or {}), **(d or {})}
    if "K" in d:
        K = np.asarray(d["K"], dtype=np.float64).reshape(3, 3)
        d.setdefault("fx", K[0, 0]); d.setdefault("fy", K[1, 1])
        d.setdefault("cx", K[0, 2]); d.setdefault("cy", K[1, 2])
    return d


def load_json(path) -> list[CameraView]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    img_dir = os.path.join(base, cfg.get("images_dir", "."))
    default_intr = _intr(cfg.get("intrinsics"))
    T_cam_lidar = np.asarray(cfg["T_cam_from_lidar"], float).reshape(4, 4) if "T_cam_from_lidar" in cfg else None
    views = []
    for i, fr in enumerate(cfg["frames"]):
        it = _intr(fr.get("intrinsics"), default_intr)
        if "T_cam_from_world" in fr:
            T = np.asarray(fr["T_cam_from_world"], float).reshape(4, 4)
        elif "T_world_from_cam" in fr:
            T = np.linalg.inv(np.asarray(fr["T_world_from_cam"], float).reshape(4, 4))
        elif "T_world_from_lidar" in fr:
            if T_cam_lidar is None:
                raise ValueError("使用 T_world_from_lidar 时需要提供全局 T_cam_from_lidar 外参")
            T = T_cam_lidar @ np.linalg.inv(np.asarray(fr["T_world_from_lidar"], float).reshape(4, 4))
        else:
            raise ValueError(f"第 {i} 帧缺少位姿")
        model = str(it.get("model", "OPENCV" if it.get("dist") else "PINHOLE")).upper()
        if model == "PINHOLE" and it.get("dist"):
            model = "OPENCV"
        img = fr["image"]
        views.append(CameraView(img, os.path.join(img_dir, img), int(it.get("width", 0)),
                                int(it.get("height", 0)), float(it["fx"]), float(it["fy"]),
                                float(it["cx"]), float(it["cy"]), model,
                                list(it.get("dist", [])), T[:3, :3].copy(), T[:3, 3].copy()))
    return views


def load_cameras(path):
    """path 可以是 JSON 文件或 COLMAP 文件夹。"""
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return load_json(path)
    model_dir, img_dir = find_colmap_dirs(path)
    if model_dir is None:
        raise FileNotFoundError("未找到 COLMAP 模型（cameras.bin/txt、images.bin/txt）")
    if img_dir is None:
        raise FileNotFoundError("找到了 COLMAP 模型，但未找到图像文件夹（images/）")
    return load_colmap(model_dir, img_dir)


# ================================================================ 投影
def _distort(x, y, model, dist):
    if model == "PINHOLE" or not dist:
        return x, y
    if model == "FISHEYE":  # 等距鱼眼
        k = list(dist) + [0] * (4 - len(dist))
        r = np.sqrt(x * x + y * y)
        th = np.arctan(r)
        th2 = th * th
        thd = th * (1 + k[0] * th2 + k[1] * th2 ** 2 + k[2] * th2 ** 3 + k[3] * th2 ** 4)
        s = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
        return x * s, y * s
    k1, k2, p1, p2, k3 = (list(dist) + [0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def _bilinear(img, u, v):
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = u.astype(np.int32), v.astype(np.int32)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    a = img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu
    b = img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu
    return a * (1 - fv) + b * fv


def colorize(xyz: np.ndarray, views: list[CameraView], progress=lambda m, p: None,
             zbuf_cell=4, depth_tol=0.03):
    """返回 (rgb uint8 (N,3), 观测权重 (N,) float32)。权重为 0 的点没有被任何相机看到。"""
    n = len(xyz)
    acc = np.zeros((n, 3), np.float64)
    wsum = np.zeros(n, np.float64)
    nv = len(views)
    for vi, cv in enumerate(views):
        progress(f"相机着色 {vi + 1}/{nv}: {cv.name}", int(100 * vi / max(nv, 1)))
        if not os.path.exists(cv.image_path):
            continue
        img = np.asarray(Image.open(cv.image_path).convert("RGB"), dtype=np.float32)
        H, W = img.shape[:2]
        sx = W / cv.width if cv.width else 1.0
        sy = H / cv.height if cv.height else 1.0

        pc = xyz @ cv.R.T + cv.t
        z = pc[:, 2]
        front = np.nonzero(z > 0.05)[0]
        if len(front) == 0:
            continue
        zf = z[front]
        x, y = pc[front, 0] / zf, pc[front, 1] / zf
        lim = 3.0 if cv.model != "FISHEYE" else 50.0  # 防止畸变多项式在视场外“折回”
        ok = (np.abs(x) < lim) & (np.abs(y) < lim)
        xd, yd = _distort(x, y, cv.model, cv.dist)
        u = (cv.fx * xd + cv.cx) * sx
        v = (cv.fy * yd + cv.cy) * sy
        ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        idx = front[ok]
        if len(idx) == 0:
            continue
        u, v, zf = u[ok], v[ok], zf[ok]

        # --- 遮挡：粗网格 z-buffer，取每个格子最近深度并做 3x3 最小值膨胀
        gw, gh = int(np.ceil(W / zbuf_cell)), int(np.ceil(H / zbuf_cell))
        pid = (v // zbuf_cell).astype(np.int64) * gw + (u // zbuf_cell).astype(np.int64)
        order = np.argsort(zf, kind="stable")
        upid, first = np.unique(pid[order], return_index=True)
        zb = np.full(gw * gh, np.inf, np.float64)
        zb[upid] = zf[order][first]
        zb = minimum_filter(zb.reshape(gh, gw), size=3, mode="nearest").ravel()
        vis = zf <= zb[pid] * (1 + depth_tol) + 0.02
        idx, u, v, zf = idx[vis], u[vis], v[vis], zf[vis]

        col = _bilinear(img, u, v)
        # 权重：越近、越靠近图像中心越可信
        rn = ((u / W - 0.5) ** 2 + (v / H - 0.5) ** 2) * 2
        w = (1.0 / np.maximum(zf, 0.1)) * np.clip(1 - 0.7 * rn, 0.05, 1)
        w = w * w  # 强调最佳视角，减少多视角平均造成的发灰
        acc[idx] += col * w[:, None]   # 同一视角中每个点只出现一次，可直接索引累加
        wsum[idx] += w
    progress("相机着色完成", 100)
    rgb = np.zeros((n, 3), np.uint8)
    seen = wsum > 0
    rgb[seen] = np.clip(acc[seen] / wsum[seen, None], 0, 255).astype(np.uint8)
    return rgb, wsum.astype(np.float32)


def inpaint_colors(xyz: np.ndarray, rgb: np.ndarray, weight: np.ndarray, k=8, max_radius=None,
                   progress=lambda m, p: None):
    """给没有被相机拍到的点补颜色，使着色连续。

    从已着色的点里找 k 个最近邻，按距离反比加权平均；
    超过 max_radius（默认 = 60 倍点间距）的点保持未着色。
    返回 (rgb, filled_mask)。filled_mask 表示“相机直接看到或已补色”的点。
    """
    from scipy.spatial import cKDTree
    seen = weight > 0
    out = rgb.copy()
    if seen.all() or not seen.any():
        return out, seen.copy()
    src = xyz[seen]
    tree = cKDTree(src)
    if max_radius is None:
        sub = src[:: max(1, len(src) // 100_000)]
        sp = np.median(cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]) * np.sqrt(max(1, len(src) // 100_000))
        max_radius = 60 * sp
    q = np.nonzero(~seen)[0]
    progress(f"补全未拍到区域的颜色：{len(q):,} 个点", 95)
    kk = int(min(k, len(src)))
    d, nb = tree.query(xyz[q], k=kk, distance_upper_bound=max_radius, workers=-1)
    if kk == 1:
        d, nb = d[:, None], nb[:, None]
    valid = np.isfinite(d)
    w = np.where(valid, 1.0 / np.maximum(d, 1e-9), 0.0)
    ok = w.sum(1) > 0
    nbc = np.where(valid[..., None], rgb[seen][np.minimum(nb, len(src) - 1)], 0).astype(np.float64)
    col = (nbc * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]
    out[q[ok]] = np.clip(col[ok], 0, 255).astype(np.uint8)
    filled = seen.copy()
    filled[q[ok]] = True
    return out, filled
