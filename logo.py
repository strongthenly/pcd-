"""程序 Logo / 启动画面（全部用代码绘制，无需图片资源）。"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QBrush, QFont)
from PySide6.QtWidgets import QSplashScreen

import theme
from version import APP_NAME, APP_NAME_EN, __version__


def _hexagon(cx, cy, r, rounding=0.12):
    pts = [QPointF(cx + r * math.sin(math.radians(60 * i)), cy - r * math.cos(math.radians(60 * i)))
           for i in range(6)]
    path = QPainterPath()
    for i in range(6):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 6]
        p1 = b + (a - b) * rounding
        p2 = b + (c - b) * rounding
        if i == 0:
            path.moveTo(p1)
        else:
            path.lineTo(p1)
        path.quadTo(b, p2)
    path.closeSubpath()
    return path


def _terrain_points(n=17):
    """一片起伏的点云曲面，投影到 2D，返回 (x, y, depth, height)。"""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = 0.32 * np.sin(2.4 * X + 0.6) * np.cos(1.9 * Y) + 0.12 * np.sin(4 * Y)
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    a, e = math.radians(-35), math.radians(-58)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    q = p @ (Rx @ Rz).T
    order = np.argsort(q[:, 2])  # 远的先画
    return q[order, 0], -q[order, 1], q[order, 2], p[order, 2]


def draw_logo(size: int = 256) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    c = size / 2
    small = size <= 48           # 小尺寸图标：点更大、去掉细节，保证清晰
    r = size * (0.46 if small else 0.47)
    hexa = _hexagon(c, c, r)

    # 底色
    bg = QLinearGradient(0, 0, 0, size)
    bg.setColorAt(0, QColor("#10213a"))
    bg.setColorAt(1, QColor("#050910"))
    p.fillPath(hexa, bg)
    glow = QRadialGradient(c, c * 1.05, r)
    glow.setColorAt(0, QColor(34, 211, 238, 70))
    glow.setColorAt(1, QColor(34, 211, 238, 0))
    p.fillPath(hexa, glow)

    # 点云曲面
    p.save()
    p.setClipPath(hexa)
    x, y, d, h = _terrain_points(9 if small else 17)
    hn = (h - h.min()) / (h.max() - h.min())
    dn = (d - d.min()) / (d.max() - d.min())
    lo, mid, hi = np.array([59, 130, 246]), np.array([34, 211, 238]), np.array([235, 255, 255])
    for xi, yi, di, hi_ in zip(x, y, dn, hn):
        col = lo + (mid - lo) * min(hi_ * 2, 1) if hi_ < 0.5 else mid + (hi - mid) * (hi_ - 0.5) * 2
        alpha = int(110 + 145 * di)
        rad = size * (0.0055 + 0.0085 * di) * (2.2 if small else 1.0)
        k = 0.68 if small else 0.5
        px, py = c + xi * r * k, c * 1.04 + yi * r * k
        if size >= 64 and hi_ > 0.82:  # 高处的点加光晕
            gg = QRadialGradient(px, py, rad * 3)
            gg.setColorAt(0, QColor(int(col[0]), int(col[1]), int(col[2]), 70))
            gg.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(gg)
            p.drawEllipse(QPointF(px, py), rad * 3, rad * 3)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(int(col[0]), int(col[1]), int(col[2]), alpha))
        p.drawEllipse(QPointF(px, py), rad, rad)
    # 扫描线
    if small:
        p.restore()
    else:
        _scan(p, size, c)
        p.restore()
    _frame(p, size, c, r, hexa, small)
    p.end()
    return img


def _scan(p, size, c):
    scan = QLinearGradient(0, c * 0.78, 0, c * 0.9)
    scan.setColorAt(0, QColor(34, 211, 238, 0))
    scan.setColorAt(1, QColor(34, 211, 238, 90))
    p.fillRect(QRectF(0, c * 0.78, size, c * 0.12), scan)
    p.setPen(QPen(QColor(150, 240, 255, 200), max(1.0, size * 0.006)))
    p.drawLine(QPointF(0, c * 0.9), QPointF(size, c * 0.9))


def _frame(p, size, c, r, hexa, small):
    ring = QLinearGradient(0, 0, size, size)
    ring.setColorAt(0, QColor(theme.ACCENT))
    ring.setColorAt(1, QColor(theme.ACCENT2))
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QBrush(ring), max(1.5, size * (0.06 if small else 0.035))))
    p.drawPath(hexa)
    if size >= 96:
        p.setPen(QPen(QColor(34, 211, 238, 80), size * 0.008))
        p.drawPath(_hexagon(c, c, r * 0.86))


def app_icon_pixmap(size=256):
    return QPixmap.fromImage(draw_logo(size))


# ------------------------------------------------------------ 启动画面
class Splash(QSplashScreen):
    W, H = 640, 420

    def __init__(self):
        super().__init__(self._make_pixmap())
        self._msg = "正在启动…"
        self._pct = 0

    def _make_pixmap(self):
        W, H = self.W, self.H
        pm = QPixmap(W, H)
        pm.fill(QColor(theme.BG0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QRadialGradient(W / 2, H * 0.32, W * 0.6)
        bg.setColorAt(0, QColor("#0f2236"))
        bg.setColorAt(1, QColor(theme.BG0))
        p.fillRect(0, 0, W, H, bg)
        # 网格
        p.setPen(QPen(QColor(34, 211, 238, 14), 1))
        for gx in range(0, W, 24):
            p.drawLine(gx, 0, gx, H)
        for gy in range(0, H, 24):
            p.drawLine(0, gy, W, gy)
        # 角标
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        L, m = 22, 12
        for (x, y, dx, dy) in [(m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            p.drawLine(x, y, x + dx * L, y)
            p.drawLine(x, y, x, y + dy * L)
        # Logo
        ls = 168
        p.drawImage(QRectF((W - ls) / 2, 34, ls, ls), draw_logo(ls * 2))
        # 标题
        f = QFont()
        f.setPixelSize(30)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#eaf6ff"))
        p.drawText(QRectF(0, 214, W, 40), Qt.AlignHCenter | Qt.AlignVCenter, APP_NAME)
        f2 = theme.mono_font(11)
        f2.setLetterSpacing(QFont.AbsoluteSpacing, 4)
        p.setFont(f2)
        p.setPen(QColor(theme.DIM))
        p.drawText(QRectF(0, 254, W, 20), Qt.AlignHCenter, APP_NAME_EN)
        # 版本号
        vf = theme.mono_font(14, bold=True)
        p.setFont(vf)
        txt = f"v{__version__}"
        tw = p.fontMetrics().horizontalAdvance(txt) + 24
        rect = QRectF((W - tw) / 2, 286, tw, 24)
        p.setPen(QPen(QColor(theme.ACCENT), 1))
        p.setBrush(QColor(34, 211, 238, 25))
        p.drawRoundedRect(rect, 3, 3)
        p.setPen(QColor(theme.ACCENT))
        p.drawText(rect, Qt.AlignCenter, txt)
        p.end()
        return pm

    def set_progress(self, msg, pct):
        self._msg, self._pct = msg, pct
        self.repaint()
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def drawContents(self, p: QPainter):
        W, H = self.W, self.H
        p.setFont(theme.mono_font(11))
        p.setPen(QColor(theme.DIM))
        p.drawText(QRectF(40, H - 62, W - 80, 18), Qt.AlignLeft, self._msg)
        p.drawText(QRectF(40, H - 62, W - 80, 18), Qt.AlignRight, f"{self._pct:3d}%")
        p.fillRect(QRectF(40, H - 40, W - 80, 2), QColor(theme.LINE))
        grad = QLinearGradient(40, 0, W - 40, 0)
        grad.setColorAt(0, QColor(theme.ACCENT2))
        grad.setColorAt(1, QColor(theme.ACCENT))
        p.fillRect(QRectF(40, H - 40, (W - 80) * self._pct / 100, 2), grad)
