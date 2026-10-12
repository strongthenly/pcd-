"""主窗口：文件加载、后台后处理、相机着色、显示参数。"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QSpinBox,
    QComboBox, QSlider, QGroupBox, QFormLayout, QFileDialog, QMessageBox, QProgressBar,
    QDockWidget, QScrollArea, QDoubleSpinBox, QDialog, QAbstractSpinBox, QTextBrowser, QDialogButtonBox,
)
from scipy.spatial import cKDTree

import colorize
import processing
import theme
from gl_view import CloudView, fmt_len
from logo import draw_logo
from pcd_io import PointCloud, read_pcd, write_pcd
from version import APP_NAME, APP_NAME_EN, __version__

HELP = (
    f"<span style='color:{theme.ACCENT}'>◢ 操作</span><br>"
    "左键拖动　水平绕 Z / 垂直俯仰<br>"
    "Shift+左键　绕 Y 轴<br>"
    "按住 X/Y/Z + 左键　锁定单轴<br>"
    "右键拖动　平移　　滚轮　缩放<br>"
    "双击　设旋转中心　R 重置　F 适配　H 隐藏HUD"
)

DATA_HELP = f"""
<h3 style='color:{theme.ACCENT}'>后处理需要哪些数据</h3>
<p><b>① 点云 .pcd（必需）</b><br>
字段至少有 <code>x y z</code>，单位默认米（可在“显示 → 数据单位”中切换）。
有 <code>intensity</code> 会提供强度着色；有 <code>rgb</code> 会作为备用颜色。
支持 ascii / binary / binary_compressed。</p>

<p><b>② 相机图像（相机着色需要）</b><br>
jpg / png 等，和点云覆盖同一区域；越多视角，着色越完整、越准确。
图像要和下面的位姿一一对应。</p>

<p><b>③ 相机内参（相机着色需要）</b><br>
fx、fy、cx、cy 和图像分辨率；有畸变时提供畸变系数
（OpenCV 的 k1 k2 p1 p2 k3，或鱼眼的 k1~k4）。</p>

<p><b>④ 每张图像拍摄时的位姿（相机着色需要）</b>，任选一种：<br>
• COLMAP 工程（sparse/0 + images）—— 图像位姿与点云必须来自同一次重建；<br>
• 每张图的相机位姿 <code>T_world_from_cam</code>（4×4，点云坐标系）；<br>
• SLAM 输出的雷达位姿 <code>T_world_from_lidar</code> + 雷达到相机外参 <code>T_cam_from_lidar</code>。<br>
图像时间戳要和位姿对齐（同一时刻），否则颜色会错位。</p>

<p><b>坐标约定</b>：相机坐标系为 OpenCV（x 右、y 下、z 前）；
位姿与点云必须在同一个坐标系、同一单位下。</p>

<p style='color:{theme.DIM}'>只有 ① 时也能用：去噪、补空隙、加密、高斯泼溅渲染都可以进行，
颜色使用高度 / 强度 / 原始 RGB。JSON 格式示例见 README.md。</p>
"""


class Worker(QThread):
    progress = Signal(str, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            res = self.fn(*self.args, lambda m, p=0: self.progress.emit(m, int(p)))
            self.done.emit(res)
        except Exception as e:  # noqa
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------ 后台任务
def job_process(raw: PointCloud, opt: processing.ProcessOptions, views, progress):
    t0 = time.time()
    pc = raw
    progress(f"读取完成：{len(pc):,} 点", 5)
    if opt.denoise:
        progress("统计滤波去噪…", 10)
        n0 = len(pc)
        pc = processing.statistical_outlier_removal(pc, opt.denoise_k, opt.denoise_std)
        progress(f"去噪：移除 {n0 - len(pc):,} 个离群点", 20)
    if opt.fill_gaps:
        progress("填补空隙…", 22)
        pc = processing.fill_gaps(pc, opt.max_gap, opt.max_points, progress=lambda m, p: progress(m, 30))
    if opt.densify_passes > 0:
        progress("稠密化…", 35)
        pc = processing.densify(pc, opt.densify_passes, opt.densify_k, opt.max_points,
                                progress=lambda m, p: progress(m, 40))
    res = {"pc": pc}
    res.update(job_gaussians(pc, opt, progress))
    if views:
        res.update(job_colorize(pc, views, progress))
    progress(f"后处理完成：{len(pc):,} 点，用时 {time.time() - t0:.1f}s", 100)
    return res


def job_gaussians(pc, opt, progress):
    cov6, normals, scales, rots, spacing = processing.estimate_gaussians(
        pc.xyz, opt.splat_k, progress=lambda m, p: progress(m, 45 + p * 0.25))
    return {"cov6": cov6, "scales": scales, "rots": rots, "spacing": spacing}


def job_colorize(pc, views, progress):
    rgb, w = colorize.colorize(pc.xyz, views, progress=lambda m, p: progress(m, 70 + p * 0.25))
    rgb, filled = colorize.inpaint_colors(pc.xyz, rgb, w, progress=progress)
    return {"cam_rgb": rgb, "cam_w": w, "cam_filled": filled}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{__version__}")
        self.setWindowIcon(QIcon(QPixmap.fromImage(draw_logo(256))))
        self.setStyleSheet(theme.QSS)
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self.view = CloudView()
        self.setCentralWidget(self.view)
        self.view.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 5000))

        self.raw: PointCloud | None = None
        self.pc: PointCloud | None = None
        self.cov6 = self.scales = self.rots = None
        self.cam_rgb = self.cam_w = self.cam_filled = None
        self.views = None
        self.path = None
        self.worker = None

        self._build_panel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("打开或拖入一个 .pcd 文件开始")

    # ------------------------------------------------------------ UI
    def _build_panel(self):
        panel = QWidget()
        panel.setObjectName("Panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 8, 10, 10)

        # 顶部：Logo + 名称 + 版本
        head = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QPixmap.fromImage(draw_logo(96)).scaled(
            44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        head.addWidget(icon)
        title = QLabel(f"<div style='font-size:15px;font-weight:bold;color:#eaf6ff'>{APP_NAME}</div>"
                       f"<div style='font-size:9px;color:{theme.DIM};letter-spacing:1px'>{APP_NAME_EN}</div>"
                       f"<div style='font-size:10px;color:{theme.ACCENT}'>v{__version__}</div>")
        head.addWidget(title, 1)
        lay.addLayout(head)

        g = QGroupBox("文件 · FILE")
        gl = QVBoxLayout(g)
        for text, fn, primary in [("打开 PCD…", self.open_pcd, True),
                                  ("加载相机（COLMAP 文件夹）…", self.open_colmap, False),
                                  ("加载相机（JSON）…", self.open_cam_json, False),
                                  ("保存处理后的 PCD…", self.save_pcd, False),
                                  ("导出 3DGS 高斯 .ply…", self.export_ply, False),
                                  ("需要哪些数据？", self.show_data_help, False)]:
            b = QPushButton(text)
            if primary:
                b.setObjectName("Primary")
            b.clicked.connect(fn)
            gl.addWidget(b)
        lay.addWidget(g)

        g = QGroupBox("后处理 · PROCESS")
        f = QFormLayout(g)
        self.cb_auto = QCheckBox("打开文件时自动处理")
        self.cb_auto.setChecked(True)
        f.addRow(self.cb_auto)
        self.cb_denoise = QCheckBox("统计滤波去噪")
        self.cb_denoise.setChecked(True)
        f.addRow(self.cb_denoise)
        self.cb_fill = QCheckBox("填补扫描线之间的空隙")
        self.cb_fill.setChecked(True)
        self.cb_fill.setToolTip("对激光雷达“线内密、线间疏”的数据，在相邻扫描线之间的表面上补点")
        f.addRow(self.cb_fill)
        self.sp_gap = QDoubleSpinBox()
        self.sp_gap.setRange(0, 100)
        self.sp_gap.setDecimals(2)
        self.sp_gap.setSingleStep(0.1)
        self.sp_gap.setSpecialValueText("自动")
        self.sp_gap.setSuffix(" m")
        self.sp_gap.setToolTip("比这更大的空洞视为真实空洞，不会被填补。0 = 自动估计")
        f.addRow("最大填补距离", self.sp_gap)
        self.sp_dense = QSpinBox()
        self.sp_dense.setRange(0, 4)
        self.sp_dense.setValue(1)
        self.sp_dense.setToolTip("邻域中点插值，每轮点数约增加 2~4 倍")
        f.addRow("加密轮数", self.sp_dense)
        self.sp_max = QDoubleSpinBox()
        self.sp_max.setRange(0.5, 50)
        self.sp_max.setValue(6.0)
        self.sp_max.setSuffix(" 百万点")
        for sp in (self.sp_gap, self.sp_dense, self.sp_max):
            sp.setButtonSymbols(QAbstractSpinBox.NoButtons)
        f.addRow("点数上限", self.sp_max)
        b = QPushButton("▶  重新处理")
        b.setObjectName("Primary")
        b.clicked.connect(self.reprocess)
        f.addRow(b)
        lay.addWidget(g)

        g = QGroupBox("显示 · RENDER")
        f = QFormLayout(g)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["高斯泼溅", "点"])
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        f.addRow("渲染", self.cmb_mode)
        self.cmb_color = QComboBox()
        self.cmb_color.currentIndexChanged.connect(self.refresh_colors)
        f.addRow("着色", self.cmb_color)
        self.sl_pt = self._slider(1, 10, 2, lambda v: self._set("point_size", float(v)))
        f.addRow("点大小", self.sl_pt)
        self.sl_sc = self._slider(20, 400, 100, lambda v: self._set("splat_scale", v / 100))
        f.addRow("泼溅尺度", self.sl_sc)
        self.sl_op = self._slider(10, 100, 95, lambda v: self._set("opacity", v / 100))
        f.addRow("不透明度", self.sl_op)
        self.cmb_bg = QComboBox()
        self.cmb_bg.addItems(["科技深色", "纯黑", "白色"])
        self.cmb_bg.currentIndexChanged.connect(self._bg_changed)
        f.addRow("背景", self.cmb_bg)
        self.cmb_unit = QComboBox()
        self.cmb_unit.addItems(["米 (m)", "厘米 (cm)", "毫米 (mm)"])
        self.cmb_unit.setToolTip("PCD 中坐标的单位，影响精度、比例尺等读数")
        self.cmb_unit.currentIndexChanged.connect(self._unit_changed)
        f.addRow("数据单位", self.cmb_unit)
        self.cb_hole = QCheckBox("连续表面（屏幕空间填洞）")
        self.cb_hole.setChecked(True)
        self.cb_hole.setToolTip("用周围颜色补齐表面上的小空洞，物体外的背景不受影响")
        self.cb_hole.toggled.connect(lambda v: self._set("hole_fill", v))
        f.addRow(self.cb_hole)
        self.sl_hole = self._slider(1, 7, 5, lambda v: self._set("hole_levels", v))
        self.sl_hole.setToolTip("填洞强度：可补齐约 2^n 像素大小的空洞")
        f.addRow("填洞强度", self.sl_hole)
        self.cb_hud = QCheckBox("显示测量 HUD（H 键切换）")
        self.cb_hud.setChecked(True)
        self.cb_hud.toggled.connect(lambda v: self._set("show_hud", v))
        f.addRow(self.cb_hud)
        lay.addWidget(g)

        g = QGroupBox("视角 · VIEW")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("左键旋转轴"))
        self.cmb_axis = QComboBox()
        self.cmb_axis.addItems(["自由", "X 轴", "Y 轴", "Z 轴"])
        self.cmb_axis.currentIndexChanged.connect(
            lambda i: (setattr(self.view, "axis_lock", [None, "x", "y", "z"][i]), self.view.update()))
        row.addWidget(self.cmb_axis)
        v.addLayout(row)
        row = QHBoxLayout()
        for text, name in [("俯视", "top"), ("正视", "front"), ("侧视", "side")]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, n=name: self.view.set_standard_view(n))
            row.addWidget(b)
        b = QPushButton("重置")
        b.clicked.connect(self.view.reset_view)
        row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("信息 · INFO")
        v = QVBoxLayout(g)
        self.lbl_info = QLabel("—")
        self.lbl_info.setObjectName("Info")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setFont(theme.mono_font(11))
        v.addWidget(self.lbl_info)
        lay.addWidget(g)
        h = QLabel(HELP)
        h.setWordWrap(True)
        h.setStyleSheet(f"color:{theme.DIM}; font-size:11px; padding:4px")
        lay.addWidget(h)
        lay.addStretch(1)

        sa = QScrollArea()
        sa.setWidget(panel)
        sa.setWidgetResizable(True)
        dock = QDockWidget("CONTROL PANEL", self)
        dock.setWidget(sa)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(320)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _slider(self, lo, hi, val, fn):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(fn)
        return s

    def _set(self, attr, val):
        setattr(self.view, attr, val)
        self.view.update()

    def _mode_changed(self, i):
        self.view.mode = "splat" if i == 0 else "points"
        self.view.update()

    def _bg_changed(self, i):
        self.view.bg_name = ["tech", "black", "white"][i]
        self.view.update()

    def _unit_changed(self, i):
        self.view.unit_scale = [1.0, 0.01, 0.001][i]
        self.view.update()
        self._update_info()

    def show_data_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("后处理数据需求")
        dlg.resize(620, 560)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(DATA_HELP)
        tb.setStyleSheet(f"background:{theme.BG2}; border:1px solid {theme.LINE}; padding:8px;")
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def options(self):
        return processing.ProcessOptions(
            denoise=self.cb_denoise.isChecked(),
            fill_gaps=self.cb_fill.isChecked(),
            max_gap=self.sp_gap.value(),
            densify_passes=self.sp_dense.value(),
            max_points=int(self.sp_max.value() * 1e6),
        )

    # ------------------------------------------------------------ 后台任务
    def run_job(self, fn, args, on_done):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "上一个任务还在运行")
            return
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker = Worker(fn, *args)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(lambda r: (self.progress.setVisible(False), on_done(r)))
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.statusBar().showMessage(msg)
        self.progress.setValue(pct)

    def _on_failed(self, msg):
        self.progress.setVisible(False)
        self.statusBar().showMessage("出错：" + msg)
        QMessageBox.critical(self, "出错", msg)

    # ------------------------------------------------------------ 文件
    def open_pcd(self):
        p, _ = QFileDialog.getOpenFileName(self, "打开点云", "", "点云 (*.pcd);;所有文件 (*)")
        if p:
            self.load_pcd(p)

    def load_pcd(self, path):
        try:
            t0 = time.time()
            self.raw = read_pcd(path)
        except Exception as e:  # noqa
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self.path = path
        self.cam_rgb = self.cam_w = self.cam_filled = None
        self.setWindowTitle(f"{APP_NAME}  v{__version__}  —  {os.path.basename(path)}")
        self.statusBar().showMessage(f"已读取 {len(self.raw):,} 点（{time.time() - t0:.1f}s）")
        # 先显示原始点云，后台处理完成后再替换
        self.pc, self.cov6, self.scales, self.rots = self.raw, None, None, None
        self.view.point_spacing = self._quick_spacing(self.raw.xyz)
        self._update_color_modes()
        self.view.set_cloud(self.raw.xyz, self.current_colors(), None)
        self._update_info()
        if self.cb_auto.isChecked():
            self.reprocess()

    def reprocess(self):
        if self.raw is None:
            return
        self.run_job(job_process, (self.raw, self.options(), self.views), self._on_processed)

    def _on_processed(self, res):
        self.pc = res["pc"]
        self.cov6, self.scales, self.rots = res["cov6"], res["scales"], res["rots"]
        self.view.point_spacing = res["spacing"]
        if "cam_rgb" in res:
            self.cam_rgb, self.cam_w, self.cam_filled = res["cam_rgb"], res["cam_w"], res["cam_filled"]
        else:
            self.cam_rgb = self.cam_w = self.cam_filled = None
        self._update_color_modes(prefer_camera="cam_rgb" in res)
        self.view.set_cloud(self.pc.xyz, self.current_colors(), self.cov6, keep_view=True)
        self._update_info()

    def open_colmap(self):
        d = QFileDialog.getExistingDirectory(self, "选择 COLMAP 工程文件夹（含 sparse/0 和 images）")
        if d:
            self.load_cameras(d)

    def open_cam_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "相机 JSON", "", "JSON (*.json)")
        if p:
            self.load_cameras(p)

    def load_cameras(self, path):
        try:
            if os.path.isdir(path):
                model_dir, img_dir = colorize.find_colmap_dirs(path)
                if model_dir and img_dir is None:
                    img_dir = QFileDialog.getExistingDirectory(self, "未自动找到图像，请选择图像文件夹")
                    if not img_dir:
                        return
                if model_dir is None:
                    raise FileNotFoundError("未找到 COLMAP 模型（cameras.bin/txt、images.bin/txt）")
                self.views = colorize.load_colmap(model_dir, img_dir)
            else:
                self.views = colorize.load_cameras(path)
        except Exception as e:  # noqa
            QMessageBox.critical(self, "加载相机失败", str(e))
            return
        missing = sum(not os.path.exists(v.image_path) for v in self.views)
        msg = f"已加载 {len(self.views)} 个相机"
        if missing:
            msg += f"（{missing} 张图像找不到，将跳过）"
        self.statusBar().showMessage(msg)
        if self.pc is not None and self.cov6 is not None:
            self.run_job(job_colorize, (self.pc, self.views), self._on_colorized)

    def _on_colorized(self, res):
        self.cam_rgb, self.cam_w, self.cam_filled = res["cam_rgb"], res["cam_w"], res["cam_filled"]
        seen = int((self.cam_w > 0).sum())
        filled = int(self.cam_filled.sum()) - seen
        self.statusBar().showMessage(
            f"相机着色完成：{seen:,}/{len(self.pc):,} 个点被相机看到，另有 {filled:,} 个点按邻域补色")
        self._update_color_modes(prefer_camera=True)
        self.refresh_colors()
        self._update_info()

    def save_pcd(self):
        if self.pc is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存 PCD", self._suggest("_processed.pcd"), "PCD (*.pcd)")
        if not p:
            return
        rgb = self.current_colors()[:, :3]
        write_pcd(p, PointCloud(self.pc.xyz, rgb, self.pc.intensity))
        self.statusBar().showMessage(f"已保存 {p}")

    def export_ply(self):
        if self.pc is None or self.scales is None:
            QMessageBox.information(self, "提示", "请先完成后处理")
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出 3DGS PLY", self._suggest("_gaussians.ply"), "PLY (*.ply)")
        if not p:
            return
        s = self.view.splat_scale
        processing.export_gaussian_ply(p, self.pc.xyz, self.current_colors()[:, :3],
                                       np.full(len(self.pc), self.view.opacity),
                                       self.scales * s, self.rots)
        self.statusBar().showMessage(f"已导出 {p}（可用 SuperSplat 等高斯泼溅查看器打开）")

    def _suggest(self, suffix):
        if not self.path:
            return ""
        return os.path.splitext(self.path)[0] + suffix

    # ------------------------------------------------------------ 着色
    def _update_color_modes(self, prefer_camera=False):
        modes = []
        if self.cam_rgb is not None:
            modes.append(("相机着色", "camera"))
        if self.pc is not None and self.pc.rgb is not None:
            modes.append(("原始 RGB", "rgb"))
        if self.pc is not None and self.pc.intensity is not None:
            modes.append(("强度", "intensity"))
        modes += [("高度 (Z)", "height"), ("单色", "white")]
        cur = self.cmb_color.currentData()
        self.cmb_color.blockSignals(True)
        self.cmb_color.clear()
        for t, k in modes:
            self.cmb_color.addItem(t, k)
        keys = [k for _, k in modes]
        if prefer_camera and "camera" in keys:
            self.cmb_color.setCurrentIndex(keys.index("camera"))
        elif cur in keys:
            self.cmb_color.setCurrentIndex(keys.index(cur))
        self.cmb_color.blockSignals(False)

    def current_colors(self) -> np.ndarray:
        pc = self.pc
        n = len(pc)
        mode = self.cmb_color.currentData() or "height"
        height = processing.scalar_colors(pc.xyz[:, 2])
        if mode == "camera" and self.cam_rgb is not None:
            fallback = pc.rgb if pc.rgb is not None else (height * 0.35 + 60).astype(np.uint8)
            mask = self.cam_filled if self.cam_filled is not None else self.cam_w > 0
            rgb = np.where(mask[:, None], self.cam_rgb, fallback)
        elif mode == "rgb" and pc.rgb is not None:
            rgb = pc.rgb
        elif mode == "intensity" and pc.intensity is not None:
            rgb = processing.scalar_colors(pc.intensity)
        elif mode == "white":
            rgb = np.full((n, 3), 220, np.uint8)
        else:
            rgb = height
        return np.hstack([rgb, np.full((n, 1), 255, np.uint8)])

    def refresh_colors(self):
        if self.pc is None:
            return
        if self.view.pos is not None and len(self.view.pos) == len(self.pc):
            self.view.set_colors(self.current_colors())

    def _update_info(self):
        if self.pc is None:
            return
        u = self.view.unit_scale
        rows = [("原始点数", f"{len(self.raw):,}"), ("当前点数", f"{len(self.pc):,}")]
        if self.view.point_spacing:
            rows.append(("点间距", fmt_len(self.view.point_spacing * u)))
        if self.views:
            rows.append(("相机", f"{len(self.views)}"))
        if self.cam_w is not None:
            rows.append(("相机覆盖", f"{(self.cam_w > 0).mean() * 100:.1f}%"))
        lo, hi = self.pc.xyz.min(0), self.pc.xyz.max(0)
        for ax, d in zip("XYZ", hi - lo):
            rows.append((f"范围 {ax}", fmt_len(d * u)))
        self.lbl_info.setText("<table width='100%'>" + "".join(
            f"<tr><td style='color:{theme.DIM}'>{k}</td><td align='right'>{v}</td></tr>" for k, v in rows)
            + "</table>")

    @staticmethod
    def _quick_spacing(xyz):
        """未处理前快速估计点间距：随机子集的最近邻距离，按面密度换算回全量。"""
        n = len(xyz)
        if n < 3:
            return None
        m = min(n, 200_000)
        sub = xyz[np.random.default_rng(0).choice(n, m, replace=False)] if m < n else xyz
        d = cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]
        return float(np.median(d) * np.sqrt(m / n))

    # ------------------------------------------------------------ 拖放
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".pcd"):
                self.load_pcd(p)
            elif p.lower().endswith(".json") or os.path.isdir(p):
                self.load_cameras(p)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(e)

    # ------------------------------------------------------------ 测试辅助
    def screenshot_when_idle(self, out, quit_after=True):
        def check():
            if self.worker and self.worker.isRunning():
                QTimer.singleShot(300, check)
                return
            QTimer.singleShot(500, lambda: self._save_shot(out, quit_after))
        QTimer.singleShot(500, check)

    def _save_shot(self, out, quit_after):
        self.view.grabFramebuffer().save(out)
        print("screenshot saved:", out)
        if quit_after:
            from PySide6.QtWidgets import QApplication
            QApplication.quit()
