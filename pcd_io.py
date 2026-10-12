"""PCD 文件读写（支持 ascii / binary / binary_compressed）。

读取结果统一为 PointCloud：
    xyz        (N,3) float64
    rgb        (N,3) uint8 或 None
    intensity  (N,)  float32 或 None
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None
    intensity: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.xyz)

    def subset(self, mask_or_idx) -> "PointCloud":
        return PointCloud(
            self.xyz[mask_or_idx],
            None if self.rgb is None else self.rgb[mask_or_idx],
            None if self.intensity is None else self.intensity[mask_or_idx],
            dict(self.meta),
        )


# ---------------------------------------------------------------- LZF
def _lzf_decompress(data: bytes, out_len: int) -> bytes:
    """LZF 解压。优先使用可选的 C 扩展 `lzf`，否则用纯 Python 实现。"""
    try:  # pip install python-lzf  (可选，加速大文件)
        import lzf  # type: ignore

        res = lzf.decompress(data, out_len)
        if res is not None:
            return res
    except ImportError:
        pass

    out = bytearray(out_len)
    ip = op = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:  # 字面量
            ln = ctrl + 1
            out[op:op + ln] = data[ip:ip + ln]
            ip += ln
            op += ln
        else:  # 回溯引用
            ln = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if ln == 7:
                ln += data[ip]
                ip += 1
            ref -= data[ip]
            ip += 1
            ln += 2
            if ref + ln <= op:  # 不重叠，可整块复制
                out[op:op + ln] = out[ref:ref + ln]
                op += ln
            else:  # 重叠（重复模式）逐字节复制
                for _ in range(ln):
                    out[op] = out[ref]
                    op += 1
                    ref += 1
    if op != out_len:
        raise ValueError(f"LZF 解压长度不符: {op} != {out_len}")
    return bytes(out)


_TYPE_MAP = {
    ("F", 4): np.float32, ("F", 8): np.float64,
    ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
    ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
}


def _parse_header(f):
    header = {}
    lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PCD 头不完整")
        s = line.decode("ascii", errors="ignore").strip()
        lines.append(s)
        if not s or s.startswith("#"):
            continue
        key, *vals = s.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    types = [v.upper() for v in header["TYPE"]]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    if "POINTS" in header:
        npts = int(header["POINTS"][0])
    else:
        npts = int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])
    return fields, sizes, types, counts, npts, header["DATA"][0].lower()


def _build_dtype(fields, sizes, types, counts):
    names, fmts = [], []
    for i, (fn, sz, tp, ct) in enumerate(zip(fields, sizes, types, counts)):
        base = np.dtype(_TYPE_MAP[(tp, sz)]).newbyteorder("<")
        name = fn if fn != "_" else f"_pad{i}"
        if name in names:
            name = f"{name}_{i}"
        names.append(name)
        fmts.append((base, (ct,)) if ct > 1 else base)
    return np.dtype({"names": names, "formats": fmts})


def read_pcd(path: str) -> PointCloud:
    with open(path, "rb") as f:
        fields, sizes, types, counts, npts, data_kind = _parse_header(f)
        dtype = _build_dtype(fields, sizes, types, counts)
        if data_kind == "ascii":
            raw = np.loadtxt(f, dtype=np.float64, ndmin=2)
            arr = np.zeros(len(raw), dtype=dtype)
            col = 0
            for name in dtype.names:
                ct = dtype[name].shape[0] if dtype[name].shape else 1
                if ct == 1 and name in ("rgb", "rgba"):
                    v = raw[:, col]
                    is_int = len(v) and np.all((v >= 0) & (v == np.floor(v))) and v.max() > 255
                    if dtype[name].kind == "f":
                        # 有的软件写整数打包值，有的写 float 位模式
                        arr[name] = (v.astype(np.uint32).view(np.float32) if is_int
                                     else v.astype(np.float32))
                    else:
                        arr[name] = v.astype(np.uint64).astype(dtype[name])
                elif ct == 1:
                    arr[name] = raw[:, col]
                else:
                    arr[name] = raw[:, col:col + ct]
                col += ct
        elif data_kind == "binary":
            buf = f.read(npts * dtype.itemsize)
            arr = np.frombuffer(buf, dtype=dtype, count=npts)
        elif data_kind == "binary_compressed":
            csize, usize = struct.unpack("<II", f.read(8))
            raw = _lzf_decompress(f.read(csize), usize)
            # 压缩数据按“列”存储：每个字段的所有点连续存放
            arr = np.zeros(npts, dtype=dtype)
            off = 0
            for name in dtype.names:
                fd = dtype[name]
                nbytes = fd.itemsize * npts
                col = np.frombuffer(raw, dtype=fd.base, count=npts * (fd.itemsize // fd.base.itemsize),
                                    offset=off)
                arr[name] = col.reshape((npts,) + fd.shape) if fd.shape else col
                off += nbytes
        else:
            raise ValueError(f"不支持的 DATA 类型: {data_kind}")

    names = arr.dtype.names
    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError("PCD 中缺少 x/y/z 字段")
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)

    rgb = None
    for key in ("rgb", "rgba"):
        if key in names:
            v = arr[key]
            u = v.view(np.uint32) if v.dtype == np.float32 else v.astype(np.uint32)
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(np.uint8)
            break
    if rgb is None and all(k in names for k in ("r", "g", "b")):
        cols = [arr[k].astype(np.float64) for k in ("r", "g", "b")]
        mx = max(c.max() for c in cols) if len(arr) else 1
        scale = 255.0 if mx <= 1.0 else 1.0
        rgb = np.clip(np.stack(cols, axis=1) * scale, 0, 255).astype(np.uint8)

    intensity = None
    for key in ("intensity", "i", "reflectivity", "scalar_intensity"):
        if key in names:
            intensity = arr[key].astype(np.float32)
            break

    good = np.isfinite(xyz).all(axis=1)
    pc = PointCloud(xyz, rgb, intensity, {"fields": fields, "data": data_kind})
    if not good.all():
        pc = pc.subset(good)
    return pc


def write_pcd(path: str, pc: PointCloud):
    """写为 binary PCD：x y z [rgb] [intensity]。"""
    n = len(pc)
    names, fmts = ["x", "y", "z"], [np.float32] * 3
    if pc.rgb is not None:
        names.append("rgb"); fmts.append(np.float32)
    if pc.intensity is not None:
        names.append("intensity"); fmts.append(np.float32)
    arr = np.zeros(n, dtype={"names": names, "formats": fmts})
    arr["x"], arr["y"], arr["z"] = pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]
    if pc.rgb is not None:
        c = pc.rgb.astype(np.uint32)
        arr["rgb"] = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).astype(np.uint32).view(np.float32)
    if pc.intensity is not None:
        arr["intensity"] = pc.intensity
    k = len(names)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {' '.join(names)}\nSIZE {' '.join(['4'] * k)}\nTYPE {' '.join(['F'] * k)}\n"
        f"COUNT {' '.join(['1'] * k)}\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
