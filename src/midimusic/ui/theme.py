"""Dark theme for the application.

A single stylesheet built from a small palette, so the accent colour is one
setting rather than a search-and-replace.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DARK", "LIGHT", "Palette", "build_stylesheet"]


@dataclass(frozen=True)
class Palette:
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_dim: str
    accent: str
    accent_text: str
    danger: str
    success: str
    warning: str


DARK = Palette(
    bg="#12131A", surface="#1A1C25", surface_alt="#22242F", border="#2E3140",
    text="#E8E9F0", text_dim="#9295A8", accent="#7C5CFF", accent_text="#FFFFFF",
    danger="#FF5C7A", success="#4ADE80", warning="#FBBF24",
)

LIGHT = Palette(
    bg="#F5F6FA", surface="#FFFFFF", surface_alt="#EDEFF5", border="#D8DBE6",
    text="#1A1C25", text_dim="#5F6478", accent="#6544E0", accent_text="#FFFFFF",
    danger="#D93B5B", success="#1F9D55", warning="#B45309",
)


def build_stylesheet(p: Palette) -> str:
    return f"""
QWidget {{
    background: {p.bg};
    color: {p.text};
    font-family: "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {p.bg}; }}

QLabel {{ background: transparent; }}
QLabel[role="title"] {{ font-size: 20px; font-weight: 600; }}
QLabel[role="subtitle"] {{ font-size: 15px; font-weight: 600; }}
QLabel[role="dim"] {{ color: {p.text_dim}; }}
QLabel[role="mono"] {{ font-family: "Cascadia Mono", "Consolas", monospace; color: {p.text_dim}; }}
QLabel[role="warn"] {{ color: {p.warning}; }}
QLabel[role="error"] {{ color: {p.danger}; }}
QLabel[role="ok"] {{ color: {p.success}; }}

QFrame[role="card"] {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame[role="divider"] {{ background: {p.border}; max-height: 1px; border: none; }}

QPushButton {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 7px 14px;
    color: {p.text};
}}
QPushButton:hover {{ background: {p.border}; }}
QPushButton:pressed {{ background: {p.surface}; }}
QPushButton:disabled {{ color: {p.text_dim}; background: {p.surface}; }}
QPushButton[role="primary"] {{
    background: {p.accent};
    border: 1px solid {p.accent};
    color: {p.accent_text};
    font-weight: 600;
    padding: 9px 20px;
}}
QPushButton[role="primary"]:hover {{ background: #8E72FF; }}
QPushButton[role="primary"]:disabled {{ background: {p.surface_alt}; border-color: {p.border}; color: {p.text_dim}; }}
QPushButton[role="danger"] {{ color: {p.danger}; }}
QPushButton[role="ghost"] {{ background: transparent; border: none; color: {p.text_dim}; padding: 4px 8px; }}
QPushButton[role="ghost"]:hover {{ color: {p.text}; background: {p.surface_alt}; }}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 7px 10px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {p.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {p.surface};
    border: 1px solid {p.border};
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
    outline: none;
}}

QSlider::groove:horizontal {{ height: 4px; background: {p.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {p.text}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {p.accent}; }}

QProgressBar {{
    background: {p.surface_alt};
    border: none; border-radius: 4px; height: 6px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 4px; }}

QTabWidget::pane {{ border: none; background: {p.bg}; }}
QTabBar::tab {{
    background: transparent; color: {p.text_dim};
    padding: 9px 18px; border: none; border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}

QListWidget, QTreeWidget, QTableWidget {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{ padding: 8px; border-radius: 6px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {p.accent}; color: {p.accent_text};
}}
QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{ background: {p.surface_alt}; }}
QHeaderView::section {{
    background: {p.surface_alt}; color: {p.text_dim};
    padding: 6px; border: none; border-bottom: 1px solid {p.border};
}}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {p.border}; border-radius: 4px; background: {p.surface_alt};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {p.accent}; border-color: {p.accent};
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p.text_dim}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QGroupBox {{
    border: 1px solid {p.border}; border-radius: 10px;
    margin-top: 14px; padding-top: 10px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {p.text_dim}; }}

QToolTip {{
    background: {p.surface_alt}; color: {p.text};
    border: 1px solid {p.border}; border-radius: 6px; padding: 6px;
}}
QSplitter::handle {{ background: {p.border}; }}
QStatusBar {{ background: {p.surface}; border-top: 1px solid {p.border}; color: {p.text_dim}; }}
QMenuBar {{ background: {p.surface}; }}
QMenuBar::item:selected {{ background: {p.surface_alt}; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 7px 24px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
"""
