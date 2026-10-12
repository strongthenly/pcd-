"""导出软件图标：assets/icon.ico（exe 图标）和 assets/logo_512.png。

    python tools/make_icon.py
"""
import io
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from logo import draw_logo  # noqa: E402


def to_pil(qimg):
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    qimg.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main():
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    out = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out, exist_ok=True)
    to_pil(draw_logo(512)).save(os.path.join(out, "logo_512.png"))
    sizes = [256, 128, 64, 48, 32, 24, 16]
    imgs = [to_pil(draw_logo(s)) for s in sizes]
    imgs[0].save(os.path.join(out, "icon.ico"), sizes=[(s, s) for s in sizes], append_images=imgs[1:])
    print("icon written to", os.path.abspath(out))


if __name__ == "__main__":
    main()
