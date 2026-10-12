"""生成一个合成演示场景：稀疏雷达点云(.pcd) + 若干相机图像 + COLMAP 文本模型 + JSON。

场景：地面(棋盘格) + 球(彩色经纬条纹) + 盒子(渐变)。相机图像通过解析光线求交渲染，
因此点云着色结果可以和真值颜色对比，检验投影/遮挡是否正确。

    python tools/make_demo.py demo_data
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcd_io import PointCloud, write_pcd  # noqa: E402

SPH_C, SPH_R = np.array([0.0, 0.0, 1.0]), 1.0
BOX_LO, BOX_HI = np.array([1.8, -1.2, 0.0]), np.array([3.0, 0.4, 1.4])


def color_of(p, obj):
    """真值颜色。obj: 0 地面, 1 球, 2 盒子"""
    c = np.zeros((len(p), 3))
    g = obj == 0
    chk = (np.floor(p[g, 0] * 1.0) + np.floor(p[g, 1] * 1.0)) % 2
    c[g] = np.where(chk[:, None] > 0, [230, 230, 220], [60, 90, 140])
    s = obj == 1
    d = p[s] - SPH_C
    lon = np.arctan2(d[:, 1], d[:, 0])
    band = (np.floor((lon + np.pi) / (np.pi / 4)) % 3).astype(int)
    c[s] = np.array([[220, 50, 50], [250, 200, 40], [40, 170, 90]])[band]
    b = obj == 2
    t = (p[b, 2] - BOX_LO[2]) / (BOX_HI[2] - BOX_LO[2])
    c[b] = np.stack([60 + 180 * t, 80 + 0 * t, 230 - 180 * t], 1)
    return c


def raycast(o, d):
    """返回 (t, obj)，未命中为 inf/-1。o:(3,) d:(N,3)"""
    n = len(d)
    best = np.full(n, np.inf)
    obj = np.full(n, -1)
    # 地面 z=0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[2] / d[:, 2]
    hit = (t > 1e-6) & (np.abs(o[0] + t * d[:, 0]) < 8) & (np.abs(o[1] + t * d[:, 1]) < 8)
    best[hit], obj[hit] = t[hit], 0
    # 球
    oc = o - SPH_C
    b = d @ oc
    cc = oc @ oc - SPH_R ** 2
    disc = b * b - cc
    with np.errstate(invalid="ignore"):
        t = -b - np.sqrt(disc)
    hit = (disc > 0) & (t > 1e-6) & (t < best)
    best[hit], obj[hit] = t[hit], 1
    # 盒子 (slab)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (BOX_LO - o) / d
        t2 = (BOX_HI - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmin > 1e-6) & (tmin < best)
    best[hit], obj[hit] = tmin[hit], 2
    return best, obj


def look_at(eye, target, up=(0, 0, 1)):
    """返回 OpenCV 约定的 world→camera 旋转 R 和平移 t"""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    R = np.stack([r, dn, f])
    return R, -R @ eye


def main(out):
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)

    # ---------- 稀疏点云：模拟一个在 (-4,-3,1.8) 的雷达多线扫描
    lidar = np.array([-4.0, -3.0, 1.8])
    az = np.linspace(-np.pi, np.pi, 1400, endpoint=False)
    el = np.radians(np.linspace(-25, 8, 24))
    A, E = np.meshgrid(az, el)
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    t, obj = raycast(lidar, dirs)
    m = np.isfinite(t)
    pts = lidar + dirs[m] * t[m, None]
    pts += rng.normal(0, 0.005, pts.shape)
    obj = obj[m]
    # 第二个雷达位置，补充另一侧
    lidar2 = np.array([5.0, 3.0, 1.8])
    t, ob2 = raycast(lidar2, dirs)
    m = np.isfinite(t)
    pts = np.vstack([pts, lidar2 + dirs[m] * t[m, None]])
    obj = np.concatenate([obj, ob2[m]])
    inten = np.where(obj == 0, 20, np.where(obj == 1, 80, 140)).astype(np.float32) + rng.normal(0, 3, len(obj))
    outl = rng.uniform([-6, -6, 0], [6, 6, 4], (300, 3))  # 离群噪声点
    pts = np.vstack([pts, outl])
    inten = np.concatenate([inten, np.full(300, 255, np.float32)])
    write_pcd(os.path.join(out, "scene.pcd"), PointCloud(pts, None, inten.astype(np.float32)))
    np.save(os.path.join(out, "_truth_obj.npy"), obj)

    # ---------- 相机
    W, H, fx = 640, 480, 520.0
    cx, cy = W / 2, H / 2
    eyes = [(-5, -4, 3), (5, -4, 3), (5, 4, 3), (-5, 4, 3), (0, -6, 2.5), (0, 6, 2.5), (-6, 0, 4), (6, 0, 4)]
    cams_txt = f"1 PINHOLE {W} {H} {fx} {fx} {cx} {cy}\n"
    img_lines, frames = [], []
    vv, uu = np.mgrid[0:H, 0:W]
    pix = np.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fx, np.ones_like(uu, float)], -1).reshape(-1, 3)
    for i, e in enumerate(eyes):
        eye = np.array(e, float)
        R, tvec = look_at(eye, np.array([0.8, -0.3, 0.6]))
        d = pix @ R  # camera → world 方向 (R^T * pix)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        tt, ob = raycast(eye, d)
        col = np.full((len(d), 3), [150, 190, 235], float)  # 天空
        hit = np.isfinite(tt)
        col[hit] = color_of(eye + d[hit] * tt[hit, None], ob[hit])
        name = f"cam_{i:02d}.png"
        Image.fromarray(col.reshape(H, W, 3).astype(np.uint8)).save(os.path.join(out, "images", name))
        # COLMAP 四元数 (wxyz)
        from scipy.spatial.transform import Rotation
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        img_lines.append(f"{i + 1} {qw} {qx} {qy} {qz} {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}\n\n")
        Twc = np.eye(4)
        Twc[:3, :3], Twc[:3, 3] = R.T, eye
        frames.append({"image": name, "T_world_from_cam": Twc.round(9).tolist()})

    with open(os.path.join(out, "sparse", "0", "cameras.txt"), "w") as f:
        f.write("# Camera list\n" + cams_txt)
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as f:
        f.write("# Image list\n" + "".join(img_lines))
    with open(os.path.join(out, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump({"images_dir": "images",
                   "intrinsics": {"width": W, "height": H, "fx": fx, "fy": fx, "cx": cx, "cy": cy},
                   "frames": frames}, f, indent=1)
    print(f"写出 {len(pts):,} 点, {len(eyes)} 个相机 → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
