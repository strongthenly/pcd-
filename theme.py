"""科技风深色主题：配色、字体、Qt 样式表。"""
from PySide6.QtGui import QColor, QFont, QFontDatabase

BG0 = "#070b12"      # 最深背景
BG1 = "#0c1320"      # 面板
BG2 = "#121c2c"      # 控件
LINE = "#1f3247"     # 分隔线
TEXT = "#cfe3f2"
DIM = "#6f8aa3"
ACCENT = "#22d3ee"   # 青色主色
ACCENT2 = "#3b82f6"  # 蓝
WARN = "#f5b942"     # 琥珀

C_ACCENT = QColor(ACCENT)
C_WARN = QColor(WARN)
C_TEXT = QColor(TEXT)
C_DIM = QColor(DIM)


def mono_font(px=12, bold=False):
    fams = set(QFontDatabase.families())
    for name in ("JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo", "Courier New"):
        if name in fams:
            f = QFont(name)
            break
    else:
        f = QFont()
        f.setStyleHint(QFont.Monospace)
    f.setPixelSize(px)
    f.setBold(bold)
    return f


QSS = f"""
* {{ color: {TEXT}; font-size: 12px; }}
QMainWindow, QWidget#Root {{ background: {BG0}; }}
QDockWidget {{ color: {DIM}; titlebar-close-icon: none; }}
QDockWidget::title {{
    background: {BG1}; padding: 6px 10px; border-bottom: 1px solid {LINE};
    text-align: left; font-weight: bold; letter-spacing: 2px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG1}; border: none; }}
QWidget#Panel {{ background: {BG1}; }}

QGroupBox {{
    background: {BG1}; border: 1px solid {LINE}; border-left: 2px solid {ACCENT};
    margin-top: 16px; padding: 10px 8px 8px 8px; border-radius: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; top: 2px; padding: 0 4px;
    color: {ACCENT}; font-weight: bold; letter-spacing: 1px;
}}

QPushButton {{
    background: {BG2}; border: 1px solid {LINE}; padding: 6px 10px; border-radius: 2px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: #ffffff; background: #15263a; }}
QPushButton:pressed {{ background: {ACCENT}; color: {BG0}; }}
QPushButton#Primary {{ border: 1px solid {ACCENT}; color: {ACCENT}; font-weight: bold; }}
QPushButton#Primary:hover {{ background: {ACCENT}; color: {BG0}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG2}; border: 1px solid {LINE}; padding: 3px 6px; border-radius: 2px;
    selection-background-color: {ACCENT}; selection-color: {BG0};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {BG2}; border: 1px solid {ACCENT}; outline: none; }}

QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {DIM}; background: {BG2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox:hover {{ color: #ffffff; }}

QSlider::groove:horizontal {{ height: 3px; background: {LINE}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; }}
QSlider::handle:horizontal {{
    width: 10px; height: 14px; margin: -6px 0; background: {BG0}; border: 2px solid {ACCENT};
}}

QStatusBar {{ background: {BG1}; color: {DIM}; border-top: 1px solid {LINE}; }}
QStatusBar QLabel {{ color: {DIM}; }}
QProgressBar {{
    background: {BG2}; border: 1px solid {LINE}; height: 10px; text-align: center; color: {TEXT};
}}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT2}, stop:1 {ACCENT}); }}

QScrollBar:vertical {{ background: {BG1}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QToolTip {{ background: {BG2}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 4px; }}
QMessageBox, QDialog {{ background: {BG1}; }}
QLabel#Section {{ color: {DIM}; letter-spacing: 1px; }}
QLabel#Info {{ color: {TEXT}; }}
"""
