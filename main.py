"""PCD 点云后处理器 入口。

用法:
    python main.py [点云.pcd] [--cameras COLMAP文件夹或相机.json] [--densify N]
                   [--mode splat|points] [--axis x|y|z] [--screenshot out.png] [--no-splash]
"""
import argparse
import sys
import time

from PySide6.QtGui import QIcon, QPixmap, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from gl_view import default_surface_format


def main():
    ap = argparse.ArgumentParser(description="PCD 点云后处理器")
    ap.add_argument("pcd", nargs="?")
    ap.add_argument("--cameras", help="COLMAP 工程文件夹或相机 JSON")
    ap.add_argument("--densify", type=int, help="加密轮数 (0-4)")
    ap.add_argument("--mode", choices=["splat", "points"])
    ap.add_argument("--axis", choices=["x", "y", "z"], help="锁定左键旋转轴")
    ap.add_argument("--screenshot", help="处理完成后截图并退出（测试用）")
    ap.add_argument("--splash-shot", help="保存启动画面截图（测试用）")
    ap.add_argument("--no-splash", action="store_true")
    args, _ = ap.parse_known_args()

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = QApplication(sys.argv)
    app.setApplicationName("PCD 点云后处理器")

    import logo
    app.setWindowIcon(QIcon(QPixmap.fromImage(logo.draw_logo(256))))
    splash = None
    t0 = time.perf_counter()
    if not args.no_splash:
        splash = logo.Splash()
        splash.show()
        splash.set_progress("初始化渲染引擎…", 10)
        if args.splash_shot:
            splash.grab().save(args.splash_shot)

    # 较重的模块在启动画面显示后再加载
    if splash:
        splash.set_progress("加载点云处理模块…", 35)
    import processing  # noqa: F401
    if splash:
        splash.set_progress("加载相机着色模块…", 60)
    import colorize  # noqa: F401
    if splash:
        splash.set_progress("构建界面…", 85)
    from app import MainWindow
    w = MainWindow()
    if args.densify is not None:
        w.sp_dense.setValue(args.densify)
    if args.mode:
        w.cmb_mode.setCurrentIndex(0 if args.mode == "splat" else 1)
    if args.axis:
        w.cmb_axis.setCurrentIndex(" xyz".index(args.axis))

    if splash:
        splash.set_progress("就绪", 100)
        while time.perf_counter() - t0 < 1.6:  # 至少显示一会，让 Logo 和版本号看得清
            app.processEvents()
            time.sleep(0.02)
    w.show()
    if splash:
        splash.finish(w)

    if args.cameras:
        w.load_cameras(args.cameras)
    if args.pcd:
        w.load_pcd(args.pcd)
    if args.screenshot:
        w.screenshot_when_idle(args.screenshot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
