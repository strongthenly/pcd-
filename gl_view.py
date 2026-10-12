"""OpenGL 点云视图：普通点 / 高斯泼溅两种渲染方式。

鼠标操作：
  左键拖动          水平 → 绕 Z 轴旋转（转台式），垂直 → 绕屏幕 X 轴俯仰
  Shift + 左键      绕 Y 轴旋转
  按住 X / Y / Z 键 + 左键拖动（或在工具栏锁定轴）→ 只绕该轴旋转
  右键 / 中键拖动    平移
  滚轮              缩放
  左键双击          以点击的点为旋转中心
  H                 显示/隐藏 HUD
"""
from __future__ import annotations

import ctypes
import math
import time

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from scipy.spatial.transform import Rotation, Slerp

import theme

POINT_VS = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uMV; uniform mat4 uP; uniform float uSize;
out vec4 vCol;
void main(){ gl_Position = uP * uMV * vec4(aPos,1.0); gl_PointSize = uSize; vCol = aCol; }
"""
POINT_FS = """#version 330 core
in vec4 vCol; out vec4 oCol;
void main(){ vec2 d = gl_PointCoord - 0.5; if (dot(d,d) > 0.25) discard; oCol = vec4(vCol.rgb, 1.0); }
"""

# 与 3D Gaussian Splatting 相同的 EWA 投影：3D 协方差 → 屏幕 2D 协方差
SPLAT_VS = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
layout(location=2) in vec3 aCovA;   // xx xy xz
layout(location=3) in vec3 aCovB;   // yy yz zz
uniform mat4 uMV; uniform mat4 uP;
uniform vec2 uFocal; uniform vec2 uTanHalf;
uniform float uScale; uniform float uOpacity; uniform float uMaxSize;
out vec4 vCol; out vec3 vConic; out float vSize;
void hide(){ gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize = 0.0; vSize = 0.0; vCol = vec4(0.0); vConic = vec3(0.0); }
void main(){
    vec4 t = uMV * vec4(aPos,1.0);
    float z = -t.z;
    if (z < 1e-4) { hide(); return; }
    gl_Position = uP * t;
    mat3 V = mat3(aCovA.x, aCovA.y, aCovA.z,
                  aCovA.y, aCovB.x, aCovB.y,
                  aCovA.z, aCovB.y, aCovB.z) * (uScale * uScale);
    mat3 W = mat3(uMV);
    mat3 S = W * V * transpose(W);
    vec2 lim = 1.3 * uTanHalf;
    float tx = clamp(t.x / z, -lim.x, lim.x) * z;
    float ty = clamp(t.y / z, -lim.y, lim.y) * z;
    mat3 J = mat3(uFocal.x / z, 0.0, 0.0,
                  0.0, uFocal.y / z, 0.0,
                  uFocal.x * tx / (z*z), uFocal.y * ty / (z*z), 0.0);
    mat3 T = J * S * transpose(J);
    float a = T[0][0] + 0.3, b = T[0][1], c = T[1][1] + 0.3;
    float det = a*c - b*b;
    if (det <= 0.0) { hide(); return; }
    vConic = vec3(c, -b, a) / det;
    float mid = 0.5 * (a + c);
    float lam = mid + sqrt(max(0.1, mid*mid - det));
    float sz = min(2.0 * ceil(3.0 * sqrt(lam)), uMaxSize);
    gl_PointSize = sz; vSize = sz;
    vCol = vec4(aCol.rgb, aCol.a * uOpacity);
}
"""
SPLAT_FS = """#version 330 core
in vec4 vCol; in vec3 vConic; in float vSize; out vec4 oCol;
void main(){
    vec2 d = (gl_PointCoord - 0.5) * vSize; d.y = -d.y;
    float power = -0.5 * (vConic.x*d.x*d.x + vConic.z*d.y*d.y) - vConic.y*d.x*d.y;
    if (power > 0.0) discard;
    float al = min(0.99, vCol.a * exp(power));
    if (al < 1.0/255.0) discard;
    oCol = vec4(vCol.rgb * al, al);   // 预乘 alpha，从后往前混合
}
"""


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def _compile(vs, fs):
    def sh(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s
    p = GL.glCreateProgram()
    for s in (sh(vs, GL.GL_VERTEX_SHADER), sh(fs, GL.GL_FRAGMENT_SHADER)):
        GL.glAttachShader(p, s)
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
    return p


def default_surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    return fmt


BG_VS = """#version 330 core
out vec2 vUV;
void main(){
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.999, 1.0);
}
"""
BG_FS = """#version 330 core
in vec2 vUV; out vec4 oCol;
uniform vec3 uTop; uniform vec3 uBottom; uniform float uVignette;
void main(){
    vec3 c = mix(uBottom, uTop, smoothstep(0.0, 1.0, vUV.y));
    vec2 d = vUV - 0.5;
    c *= 1.0 - uVignette * dot(d, d) * 1.6;
    oCol = vec4(c, 1.0);
}
"""

# 屏幕空间填洞（pull-push）：mip 第 L 级 = 2^L×2^L 像素块的预乘颜色平均。
# 对没被完全覆盖的像素，从细到粗逐级取“周围有效像素的平均色”补上；
# 只有当周围覆盖率足够高（洞被表面包围）时才补，避免糊到物体外面的背景上。
FILL_FS = """#version 330 core
in vec2 vUV; out vec4 oCol;
uniform sampler2D uTex; uniform int uLevels; uniform float uThresh;
void main(){
    vec4 acc = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
    for (int L = 1; L <= 8; ++L) {
        if (L > uLevels || acc.a >= 0.995) break;
        vec4 m = textureLod(uTex, vUV, float(L));
        if (m.a < 1e-4) continue;
        float w = smoothstep(uThresh * 0.6, uThresh, m.a);
        acc += (1.0 - acc.a) * (m / m.a) * w;
    }
    oCol = acc;
}
"""


def fmt_len(m: float) -> str:
    """把以米为单位的长度格式化为合适的单位（3 位有效数字）。"""
    a = abs(m)
    if a == 0 or not math.isfinite(a):
        return "--"
    for unit, k in (("km", 1e3), ("m", 1.0), ("cm", 1e-2), ("mm", 1e-3), ("µm", 1e-6)):
        if a >= k or unit == "µm":
            v = m / k
            return f"{v:.3g} {unit}" if abs(v) < 1000 else f"{v:,.0f} {unit}"
    return f"{m:.3g} m"


def _nice(x):
    """不大于 x 的 1/2/5×10^n。"""
    if x <= 0:
        return 0
    e = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10 ** e <= x:
            return m * 10 ** e
    return 10 ** e


BGS = {  # 背景: (顶部, 底部, 暗角, 是否浅色)
    "tech": ((0.05, 0.09, 0.14), (0.012, 0.018, 0.03), 0.55, False),
    "black": ((0, 0, 0), (0, 0, 0), 0.0, False),
    "white": ((1, 1, 1), (0.9, 0.92, 0.95), 0.0, True),
}


class CloudView(QOpenGLWidget):
    statusMessage = Signal(str)
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        # 数据
        self.pos = None          # (N,3) float32，已减去 origin
        self.rgba = None         # (N,4) uint8
        self.cov6 = None         # (N,6) float32
        self.origin = np.zeros(3)
        self._dirty = {"pos": False, "col": False, "cov": False}
        self.point_spacing = None   # 点云典型点间距（数据单位）
        self.unit_scale = 1.0       # 1 个数据单位 = 多少米
        # 渲染参数
        self.mode = "splat"
        self.point_size = 2.0
        self.splat_scale = 1.0
        self.opacity = 0.95
        self.bg_name = "tech"
        self.show_hud = True
        self.hole_fill = True       # 屏幕空间填洞，让表面连续
        self.hole_levels = 5        # 最大可填补约 2^levels 像素的空洞
        self._fbo = None
        self._fbo_size = None
        # 相机
        self.fov = 50.0
        self.R = np.eye(3)
        self.pivot = np.zeros(3)
        self.dist = 10.0
        self.pan = np.zeros(2)
        self.radius = 1.0
        self.axis_lock = None
        self._held_axis = None
        self._last = None
        self._btn = None
        self._drag_text = None      # 拖动时的读数
        self._drag_acc = None
        self._sorted_R = None
        self._gl_ok = False
        self._max_pt = 256.0
        # 动画
        self._target_dist = None
        self._view_anim = None      # (R0, R1, d0, d1, t0, dur)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._frame_ms = 0.0
        self._flash_until = 0.0     # 拖动结束后旋转中心标记保留一会
        self.R = _rot("x", -62) @ _rot("z", -35)
        self.fit_view(animate=False)

    # ------------------------------------------------------------ 数据接口
    def set_cloud(self, xyz, rgba, cov6, keep_view=False):
        self.origin = xyz.mean(axis=0) if len(xyz) else np.zeros(3)
        self.pos = np.ascontiguousarray((xyz - self.origin).astype(np.float32))
        self.rgba = np.ascontiguousarray(rgba.astype(np.uint8))
        self.cov6 = None if cov6 is None else np.ascontiguousarray(cov6.astype(np.float32))
        self._dirty = {"pos": True, "col": True, "cov": cov6 is not None}
        self._sorted_R = None
        if len(self.pos):
            lo, hi = np.percentile(self.pos, [1, 99], axis=0)
            self.radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)
        if not keep_view:
            self.fit_view(animate=False)
        self.update()

    def set_colors(self, rgba):
        self.rgba = np.ascontiguousarray(rgba.astype(np.uint8))
        self._dirty["col"] = True
        self.update()

    # ------------------------------------------------------------ 视角
    def _animate_to(self, R1, d1=None, dur=0.35):
        self._view_anim = (self.R.copy(), np.asarray(R1, float), self.dist,
                           self.dist if d1 is None else d1, time.perf_counter(), dur)
        self._target_dist = None
        self._timer.start()

    def reset_view(self):
        self.pivot = np.zeros(3)
        self.pan = np.zeros(2)
        self._animate_to(_rot("x", -62) @ _rot("z", -35), self._fit_dist())

    def _fit_dist(self):
        return self.radius / math.sin(math.radians(self.fov / 2)) * 1.05

    def fit_view(self, animate=True):
        self.pivot = np.zeros(3)
        self.pan = np.zeros(2)
        if animate:
            self._animate_to(self.R, self._fit_dist())
        else:
            self._target_dist = None
            self._view_anim = None
            self.dist = self._fit_dist()
        self.update()

    def set_standard_view(self, name):
        views = {"top": np.eye(3), "front": _rot("x", -90), "side": _rot("x", -90) @ _rot("z", -90)}
        self._animate_to(views[name])

    def _tick(self):
        busy = False
        if self._view_anim is not None:
            R0, R1, d0, d1, t0, dur = self._view_anim
            u = min(1.0, (time.perf_counter() - t0) / dur)
            u = u * u * (3 - 2 * u)  # smoothstep
            q = Slerp([0, 1], Rotation.from_matrix([R0, R1]))(u)
            self.R = q.as_matrix()
            self.dist = d0 * (d1 / d0) ** u
            if u >= 1.0:
                self.R = R1
                self._view_anim = None
            else:
                busy = True
        if self._target_dist is not None:
            self.dist += (self._target_dist - self.dist) * 0.3
            if abs(self.dist / self._target_dist - 1) < 1e-3:
                self.dist = self._target_dist
                self._target_dist = None
            else:
                busy = True
        if time.perf_counter() < self._flash_until:
            busy = True
        elif self._drag_text and self._btn is None:
            self._drag_text = None
        if not busy:
            self._timer.stop()
        self.update()

    def model_view(self):
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = -self.R @ self.pivot
        V = np.eye(4)
        V[:3, 3] = [self.pan[0], self.pan[1], -self.dist]
        return V @ M

    def projection(self, w, h):
        aspect = w / max(h, 1)
        near = max(self.dist * 0.002, self.radius * 1e-4)
        far = self.dist + self.radius * 4 + np.linalg.norm(self.pivot)
        f = 1.0 / math.tan(math.radians(self.fov) / 2)
        P = np.zeros((4, 4))
        P[0, 0], P[1, 1] = f / aspect, f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = 2 * far * near / (near - far)
        P[3, 2] = -1
        return P

    def pixel_size(self):
        """旋转中心深度处，屏幕上 1 个像素对应的实际长度（数据单位）。"""
        return 2 * self.dist * math.tan(math.radians(self.fov) / 2) / max(self.height(), 1)

    # ------------------------------------------------------------ OpenGL
    def initializeGL(self):
        try:
            self.prog_pt = _compile(POINT_VS, POINT_FS)
            self.prog_sp = _compile(SPLAT_VS, SPLAT_FS)
            self.prog_bg = _compile(BG_VS, BG_FS)
            self.prog_fill = _compile(BG_VS, FILL_FS)
        except Exception as e:  # noqa
            self.statusMessage.emit(f"着色器编译失败: {e}")
            raise
        self.vao = GL.glGenVertexArrays(1)
        self.vao_bg = GL.glGenVertexArrays(1)
        self.vbo_pos, self.vbo_col, self.vbo_cov, self.ebo = GL.glGenBuffers(4)
        try:
            self._max_pt = float(min(256.0, GL.glGetFloatv(GL.GL_POINT_SIZE_RANGE)[1]))
        except Exception:  # noqa
            self._max_pt = 64.0
        self._gl_ok = True
        self._idx_count = 0
        for k in self._dirty:
            self._dirty[k] = self._dirty[k] or self.pos is not None

    def _upload(self):
        GL.glBindVertexArray(self.vao)
        if self._dirty["pos"] and self.pos is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            self._dirty["pos"] = False
        if self._dirty["col"] and self.rgba is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_col)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.rgba.nbytes, self.rgba, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 0, ctypes.c_void_p(0))
            self._dirty["col"] = False
        if self._dirty["cov"] and self.cov6 is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_cov)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.cov6.nbytes, self.cov6, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(2)
            GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(3)
            GL.glVertexAttribPointer(3, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))
            self._dirty["cov"] = False
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _sort_splats(self, MV):
        """按视线深度从远到近排序（只与旋转有关，平移/缩放不改变顺序）。"""
        if self._sorted_R is not None and np.allclose(self._sorted_R, self.R, atol=1e-4):
            return
        # 超大点云拖动时不每帧排序（松开鼠标后再排序），保证流畅
        if self._sorted_R is not None and (self._btn is not None or self._view_anim) \
                and len(self.pos) > 2_000_000:
            return
        row = MV[2, :3].astype(np.float32)
        d = self.pos @ row
        lo, hi = float(d.min()), float(d.max())
        q = ((d - lo) * (65535.0 / max(hi - lo, 1e-9))).astype(np.uint16)
        idx = np.argsort(q, kind="stable").astype(np.uint32)   # uint16 → 基数排序，很快
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_DYNAMIC_DRAW)
        self._idx_count = len(idx)
        self._sorted_R = self.R.copy()

    def _uniform_mat(self, prog, name, m):
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(prog, name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, dtype=np.float32))

    def _draw_background(self):
        top, bottom, vig, _ = BGS[self.bg_name]
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glUseProgram(self.prog_bg)
        GL.glUniform3f(GL.glGetUniformLocation(self.prog_bg, "uTop"), *top)
        GL.glUniform3f(GL.glGetUniformLocation(self.prog_bg, "uBottom"), *bottom)
        GL.glUniform1f(GL.glGetUniformLocation(self.prog_bg, "uVignette"), vig)
        GL.glBindVertexArray(self.vao_bg)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def _draw_cloud(self, w, h, dpr):
        self._upload()
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        MV, P = self.model_view(), self.projection(w, h)
        use_splat = self.mode == "splat" and self.cov6 is not None
        GL.glBindVertexArray(self.vao)
        if use_splat:
            self._sort_splats(MV)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            p = self.prog_sp
            GL.glUseProgram(p)
            self._uniform_mat(p, "uMV", MV)
            self._uniform_mat(p, "uP", P)
            GL.glUniform2f(GL.glGetUniformLocation(p, "uFocal"), P[0, 0] * w / 2, P[1, 1] * h / 2)
            GL.glUniform2f(GL.glGetUniformLocation(p, "uTanHalf"), 1 / P[0, 0], 1 / P[1, 1])
            GL.glUniform1f(GL.glGetUniformLocation(p, "uScale"), self.splat_scale)
            GL.glUniform1f(GL.glGetUniformLocation(p, "uOpacity"), self.opacity)
            GL.glUniform1f(GL.glGetUniformLocation(p, "uMaxSize"), self._max_pt)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glDrawElements(GL.GL_POINTS, self._idx_count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glDisable(GL.GL_BLEND)
        else:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            p = self.prog_pt
            GL.glUseProgram(p)
            self._uniform_mat(p, "uMV", MV)
            self._uniform_mat(p, "uP", P)
            GL.glUniform1f(GL.glGetUniformLocation(p, "uSize"), self.point_size * dpr)
            GL.glDrawArrays(GL.GL_POINTS, 0, len(self.pos))
            GL.glDisable(GL.GL_DEPTH_TEST)

    def _ensure_fbo(self, w, h):
        if self._fbo is not None and self._fbo_size == (w, h):
            return
        if self._fbo is not None:
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures([self._fbo_tex])
            GL.glDeleteRenderbuffers(1, [self._fbo_depth])
        self._fbo_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        self._fbo_depth = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._fbo_depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._fbo_tex, 0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self._fbo_depth)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            self.hole_fill = False   # 显卡不支持时自动退回直接渲染
            self.statusMessage.emit("显卡不支持离屏渲染，已关闭填洞")
        self._fbo_size = (w, h)

    def _effective_hole_levels(self):
        """放大时点与点在屏幕上的间隔变大，相应提高填洞级别，保证近看也连续。"""
        lv = float(self.hole_levels)
        if self.point_spacing:
            ratio = self.point_spacing / max(self.pixel_size(), 1e-12)
            lv += float(np.clip(np.log2(max(ratio, 1.0)), 0, 3))
        return int(min(8, round(lv)))

    def paintGL(self):
        t_start = time.perf_counter()
        dpr = self.devicePixelRatioF()
        w, h = int(self.width() * dpr), int(self.height() * dpr)
        default_fbo = self.defaultFramebufferObject()
        has_cloud = self.pos is not None and len(self.pos)
        if has_cloud and self.hole_fill and self.hole_levels > 0:
            # 1) 点云渲染到离屏纹理（透明背景，预乘 alpha）
            self._ensure_fbo(w, h)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._draw_cloud(w, h, dpr)
            # 2) 生成 mip 金字塔：每一级是上一级的覆盖加权平均（pull）
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            # 3) 回到屏幕：画背景，再用金字塔补洞后合成（push）
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)
            GL.glViewport(0, 0, w, h)
            self._draw_background()
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            p = self.prog_fill
            GL.glUseProgram(p)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            GL.glUniform1i(GL.glGetUniformLocation(p, "uTex"), 0)
            GL.glUniform1i(GL.glGetUniformLocation(p, "uLevels"), self._effective_hole_levels())
            GL.glUniform1f(GL.glGetUniformLocation(p, "uThresh"), 0.42)
            GL.glBindVertexArray(self.vao_bg)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glDisable(GL.GL_BLEND)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        else:
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0, 0, 0, 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._draw_background()
            if has_cloud:
                self._draw_cloud(w, h, dpr)
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glFinish()
        ms = (time.perf_counter() - t_start) * 1000
        self._frame_ms = ms if not self._frame_ms else 0.8 * self._frame_ms + 0.2 * ms
        if self.show_hud:
            self._draw_hud()

    # ------------------------------------------------------------ HUD
    def _hud_colors(self):
        light = BGS[self.bg_name][3]
        if light:
            return QColor(0, 120, 150), QColor(40, 55, 70), QColor(110, 125, 140), QColor(200, 120, 0)
        return QColor(theme.ACCENT), QColor(theme.TEXT), QColor(theme.DIM), QColor(theme.WARN)

    def _draw_hud(self):
        W, H = self.width(), self.height()
        acc, txt, dim, warn = self._hud_colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 视口角标
        p.setPen(QPen(acc, 1.5))
        L, m = 18, 8
        for (x, y, dx, dy) in [(m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))

        # 左上：测量面板
        unit = self.unit_scale
        px_m = self.pixel_size() * unit
        sp_m = None if self.point_spacing is None else self.point_spacing * unit
        rows = [("视距", fmt_len(self.dist * unit), txt),
                ("屏幕精度", f"1 px = {fmt_len(px_m)}", acc),
                ("点间距", "计算中…" if sp_m is None else fmt_len(sp_m), txt)]
        if sp_m:
            ratio = sp_m / px_m       # 一个点间距在屏幕上占多少像素
            if ratio > 3:
                rows.append(("状态", f"⚠ 超出点云精度 ×{ratio:.1f}", warn))
            elif ratio > 1:
                rows.append(("状态", "接近点云精度", acc))
            else:
                rows.append(("状态", "精度正常", dim))
        npts = 0 if self.pos is None else len(self.pos)
        rows.append(("点数", f"{npts:,}", txt))
        rows.append(("渲染耗时", f"{self._frame_ms:.1f} ms", dim))
        self._panel(p, 18, 18, "MEASURE", rows, acc, dim)

        # 右上：模式标签
        mode = "GAUSSIAN SPLAT" if (self.mode == "splat" and self.cov6 is not None) else "POINT CLOUD"
        lock = self._held_axis or self.axis_lock
        tag = mode + (f"  ·  LOCK {lock.upper()}" if lock else "")
        f = theme.mono_font(11, bold=True)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(tag) + 20
        r = QRectF(W - 18 - tw, 18, tw, 22)
        p.setPen(QPen(warn if lock else acc, 1))
        fill = QColor(warn if lock else acc)
        fill.setAlpha(28)
        p.setBrush(fill)
        p.drawRect(r)
        p.setPen(warn if lock else acc)
        p.drawText(r, Qt.AlignCenter, tag)

        # 右下：比例尺
        self._scale_bar(p, W, H, px_m, acc, txt)
        # 左下：坐标轴
        self._gizmo(p, 72, H - 72, lock, dim)
        # 拖动时：旋转中心标记 + 读数
        if self._btn is not None or time.perf_counter() < self._flash_until:
            self._pivot_marker(p, W, H, acc)
        if self._drag_text:
            f = theme.mono_font(13, bold=True)
            p.setFont(f)
            tw = p.fontMetrics().horizontalAdvance(self._drag_text) + 28
            r = QRectF((W - tw) / 2, H - 58, tw, 28)
            bgc = QColor(0, 0, 0, 150) if not BGS[self.bg_name][3] else QColor(255, 255, 255, 200)
            p.setPen(QPen(acc, 1))
            p.setBrush(bgc)
            p.drawRect(r)
            p.setPen(acc)
            p.drawText(r, Qt.AlignCenter, self._drag_text)
        p.end()

    def _panel(self, p, x, y, title, rows, acc, dim):
        fv = theme.mono_font(12)
        fl = QFont()
        fl.setPixelSize(12)
        p.setFont(fv)
        lh = 19
        w = 250
        h = 26 + lh * len(rows) + 6
        dark = not BGS[self.bg_name][3]
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(4, 10, 18, 165) if dark else QColor(255, 255, 255, 200))
        p.drawRect(QRectF(x, y, w, h))
        p.setPen(QPen(acc, 2))
        p.drawLine(QPointF(x, y), QPointF(x, y + h))
        ft = theme.mono_font(10, bold=True)
        ft.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        p.setFont(ft)
        p.setPen(acc)
        p.drawText(QPointF(x + 10, y + 16), "◢ " + title)
        p.setPen(QPen(QColor(acc.red(), acc.green(), acc.blue(), 60), 1))
        p.drawLine(QPointF(x + 10, y + 22), QPointF(x + w - 10, y + 22))
        for i, (k, v, col) in enumerate(rows):
            yy = y + 26 + lh * (i + 1) - 5
            p.setFont(fl)
            p.setPen(dim)
            p.drawText(QPointF(x + 10, yy), k)
            p.setFont(fv)
            p.setPen(col)
            p.drawText(QRectF(x + 80, yy - 14, w - 90, 18), Qt.AlignRight | Qt.AlignVCenter, v)

    def _scale_bar(self, p, W, H, px_m, acc, txt):
        target_px = 140
        L = _nice(px_m * target_px)
        if L <= 0:
            return
        bar = L / px_m
        x1, y = W - 26, H - 30
        x0 = x1 - bar
        p.setPen(QPen(acc, 2))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        for i in range(6):  # 刻度
            xx = x0 + bar * i / 5
            tick = 9 if i in (0, 5) else 5
            p.drawLine(QPointF(xx, y), QPointF(xx, y - tick))
        p.setFont(theme.mono_font(12, bold=True))
        p.setPen(txt)
        p.drawText(QRectF(x0 - 40, y - 30, bar + 40, 18), Qt.AlignRight | Qt.AlignVCenter, fmt_len(L))
        p.setFont(theme.mono_font(10))
        p.setPen(QColor(txt.red(), txt.green(), txt.blue(), 140))
        p.drawText(QRectF(x0 - 10, y - 30, 20, 18), Qt.AlignCenter, "0")

    def _gizmo(self, p, cx, cy, lock, dim):
        R = 40
        dark = not BGS[self.bg_name][3]
        p.setPen(QPen(QColor(dim.red(), dim.green(), dim.blue(), 90), 1))
        p.setBrush(QColor(4, 10, 18, 140) if dark else QColor(255, 255, 255, 170))
        p.drawEllipse(QPointF(cx, cy), R + 12, R + 12)
        axes = [("X", np.array([1, 0, 0]), QColor(255, 84, 104)),
                ("Y", np.array([0, 1, 0]), QColor(120, 230, 100)),
                ("Z", np.array([0, 0, 1]), QColor(70, 160, 255))]
        items = []
        for name, a, col in axes:
            v = self.R @ a
            items.append((v[2], name, v, col, True))
            items.append((-v[2], name, -v, col, False))
        f = theme.mono_font(11, bold=True)
        p.setFont(f)
        for _, name, v, col, pos in sorted(items, key=lambda t: t[0]):
            end = QPointF(cx + v[0] * R, cy - v[1] * R)
            locked = lock == name.lower()
            if pos:
                if locked:
                    glow = QColor(col)
                    glow.setAlpha(70)
                    p.setPen(QPen(glow, 9, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(QPointF(cx, cy), end)
                p.setPen(QPen(col, 3.5 if locked else 2.2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(cx, cy), end)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(end, 8, 8)
                p.setPen(QColor(10, 14, 20))
                p.drawText(QRectF(end.x() - 8, end.y() - 8, 16, 16), Qt.AlignCenter, name)
            else:
                faint = QColor(col)
                faint.setAlpha(110)
                p.setPen(QPen(faint, 1.2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(end, 4, 4)

    def _pivot_marker(self, p, W, H, acc):
        P = self.projection(W, H)
        sx = W / 2 + P[0, 0] * self.pan[0] / self.dist * W / 2
        sy = H / 2 - P[1, 1] * self.pan[1] / self.dist * H / 2
        c = QPointF(sx, sy)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(acc, 1.5))
        p.drawEllipse(c, 12, 12)
        p.setPen(QPen(QColor(acc.red(), acc.green(), acc.blue(), 90), 1))
        p.drawEllipse(c, 22, 22)
        p.setPen(QPen(acc, 1.5))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(sx + dx * 16, sy + dy * 16), QPointF(sx + dx * 28, sy + dy * 28))
        p.setBrush(acc)
        p.drawEllipse(c, 2.2, 2.2)

    # ------------------------------------------------------------ 交互
    def mousePressEvent(self, e):
        self._last = e.position()
        self._btn = e.button()
        self._view_anim = None
        self._drag_acc = np.zeros(2)
        self.setCursor(Qt.ClosedHandCursor if self._btn != Qt.LeftButton else Qt.SizeAllCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        self._btn = None
        self.setCursor(Qt.CrossCursor)
        self._flash_until = time.perf_counter() + 0.9
        self._timer.start()
        self.update()

    def mouseMoveEvent(self, e):
        if self._last is None or self._btn is None:
            return
        p = e.position()
        dx, dy = p.x() - self._last.x(), p.y() - self._last.y()
        self._last = p
        if self._btn == Qt.LeftButton:
            s = 0.35  # 度/像素
            lock = self._held_axis or self.axis_lock
            if lock == "x":
                self.R = self.R @ _rot("x", dy * s)
                self._drag_acc[0] += dy * s
                self._drag_text = f"ROTATE  X {self._drag_acc[0]:+7.1f}°"
            elif lock in ("y", "z"):
                self.R = self.R @ _rot(lock, dx * s)
                self._drag_acc[0] += dx * s
                self._drag_text = f"ROTATE  {lock.upper()} {self._drag_acc[0]:+7.1f}°"
            elif e.modifiers() & Qt.ShiftModifier:
                self.R = self.R @ _rot("y", dx * s)
                self._drag_acc[0] += dx * s
                self._drag_text = f"ROTATE  Y {self._drag_acc[0]:+7.1f}°"
            else:
                self.R = _rot("x", dy * s) @ (self.R @ _rot("z", dx * s))
                self._drag_acc += (dx * s, dy * s)
                self._drag_text = (f"ROTATE  Z {self._drag_acc[0]:+7.1f}°   "
                                   f"PITCH {self._drag_acc[1]:+7.1f}°")
            self.R = _orthonormalize(self.R)
        elif self._btn in (Qt.RightButton, Qt.MiddleButton):
            k = self.pixel_size()
            self.pan += np.array([dx * k, -dy * k])
            self._drag_acc += (dx * k, -dy * k)
            u = self.unit_scale
            self._drag_text = (f"PAN  ΔX {fmt_len(self._drag_acc[0] * u)}   "
                               f"ΔY {fmt_len(self._drag_acc[1] * u)}")
        self.viewChanged.emit()
        self.update()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        base = self._target_dist if self._target_dist is not None else self.dist
        self._target_dist = float(np.clip(base * (0.85 ** steps), self.radius * 1e-5, self.radius * 200))
        self._view_anim = None
        self._drag_text = f"ZOOM  1 px = {fmt_len(self._target_dist / self.dist * self.pixel_size() * self.unit_scale)}"
        self._flash_until = time.perf_counter() + 0.9
        self._timer.start()

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton or self.pos is None or not len(self.pos):
            return
        w, h = self.width(), self.height()
        MVP = self.projection(w, h) @ self.model_view()
        clip = self.pos @ MVP[:, :3].T.astype(np.float32) + MVP[:, 3].astype(np.float32)
        wv = clip[:, 3]
        front = wv > 1e-6
        sx = (clip[:, 0] / np.where(front, wv, 1) * 0.5 + 0.5) * w
        sy = (1 - (clip[:, 1] / np.where(front, wv, 1) * 0.5 + 0.5)) * h
        px, py = e.position().x(), e.position().y()
        near = front & (np.abs(sx - px) < 6) & (np.abs(sy - py) < 6)
        if not near.any():
            return
        cand = np.nonzero(near)[0]
        i = cand[np.argmin(wv[cand])]
        # 以该点为新旋转中心，并保持该点在屏幕上的位置不变
        old = self.model_view() @ np.append(self.pos[i], 1.0)
        self.pivot = self.pos[i].astype(np.float64)
        self.dist = float(-old[2])
        self.pan = np.array([old[0], old[1]])
        self.statusMessage.emit("旋转中心: (%.3f, %.3f, %.3f)" % tuple(self.pivot + self.origin))
        self._flash_until = time.perf_counter() + 1.2
        self._timer.start()
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if not e.isAutoRepeat() and k in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z):
            self._held_axis = {Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z"}[k]
            self.update()
        elif k == Qt.Key_R:
            self.reset_view()
        elif k == Qt.Key_F:
            self.fit_view()
        elif k == Qt.Key_H:
            self.show_hud = not self.show_hud
            self.update()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat() and e.key() in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z):
            self._held_axis = None
            self.update()
        else:
            super().keyReleaseEvent(e)
