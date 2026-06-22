"""Virturoid Studio - a standalone native desktop app over the build/evaluate engine.

This is the product UI, not a browser preview: a real PySide6 desktop studio with a central 3D
MuJoCo viewport, resizable work panels, native menus, keyboard shortcuts, and worker-threaded
build/train/evaluate jobs so the interface stays responsive while the robot loop runs.

    python -m virturoid.desktop            # launch
    python -m virturoid.desktop --workspace build/desktop

Panels: Design Brief (left) / Simulation Viewport (center) / Inspector (right).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from PySide6.QtCore import Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QImage, QKeySequence, QPalette, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
        QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
        QSlider, QSplitter, QStackedWidget, QStyleFactory, QTabWidget, QToolButton,
        QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:  # pragma: no cover - helpful message instead of a stack trace
    sys.stderr.write("Virturoid Desktop needs PySide6.  Install it with:\n\n    pip install PySide6-Essentials\n\n")
    raise

# ---- design tokens: native robotics studio, not a web dashboard ----
# A dense pro-tool palette: graphite surfaces, signal-orange primary actions, teal telemetry, and crisp
# 1px borders. The 3D viewport stays dark so the simulated robot has contrast and depth.
BG, SURFACE, SURFACE2, BORDER = "#0b0d0f", "#15181b", "#1d2226", "#2b3238"
ELEV, BORDER2, RAIL = "#242a30", "#3b444c", "#080a0c"
SOFT = "#1c2126"   # subtle card border (barely above SURFACE) — calm, Obsidian-style elevation, not hard boxes
TEXT, MUTED, FAINT = "#f1f3ed", "#a7b0aa", "#737d76"
ACCENT, ACCENT2, ACCENT_INK = "#d97828", "#352314", "#1b1008"
SIGNAL = "#38b7ad"
OK, WARN, BAD, INFO = "#45c782", "#d69a27", "#e25a67", "#62a8ff"
# Typography: mono for instrument readouts, clean system sans for long-form controls/chat.
MONO = "Consolas"
SANS = "Segoe UI"

# How each design source is described in the Properties panel - honest per-source labels (not just llm vs. else).
_DESIGN_SOURCE_LABEL = {
    "llm": "LLM-designed", "real": "real production model",
    "imported": "imported model", "heuristic": "offline template",
}

EXAMPLES = [
    "a tabletop arm that sorts red and blue blocks into matching bins",
    "a humanoid that picks up boxes and places them on a shelf",
    "a mobile robot that carries parts across the room",
]
# Short welcome chips -> the full prompt they insert (keeps the welcome tidy on narrow panes).
SUGGESTIONS = [
    # Lead with PICK-AND-PLACE (single object) — it runs the REAL contact grasp (no pin), the strongest + most
    # 1X-relevant arm moment. The multi-object "sort" is still typeable and honestly discloses its idealized pin.
    ("Pick-and-place arm", "a tabletop arm that picks up a box and places it on a target"),
    ("Box-stacking humanoid", EXAMPLES[1]),
    ("Parts-carrying rover", EXAMPLES[2]),
    ("A frog robot that runs through a maze", "a frog-like robot that runs through a maze to the goal"),
]


def _qss() -> str:
    return f"""
    QWidget {{ background: {BG}; color: {TEXT}; font-family: {SANS}; font-size: 12px; }}
    QLabel {{ background: transparent; }}
    QMenuBar {{ background: {RAIL}; color: {MUTED}; border-bottom: 1px solid {BORDER}; padding: 2px 6px; }}
    QMenuBar::item {{ background: transparent; padding: 5px 9px; border-radius: 4px; }}
    QMenuBar::item:selected {{ background: {SURFACE2}; color: {TEXT}; }}
    QMenu {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER2}; padding: 5px; }}
    QMenu::item {{ padding: 6px 28px 6px 18px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {ACCENT2}; color: {TEXT}; }}
    QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 6px; }}
    QMainWindow::separator {{ background: {BORDER}; width: 4px; height: 4px; }}
    QMainWindow::separator:hover {{ background: {ACCENT}; }}
    QSplitter::handle {{ background: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QSplitter::handle:hover {{ background: {ACCENT}; }}

    /* ---- command bar + brand (desktop studio identity) ---- */
    QWidget#topbar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
    QLabel#brand {{ color: {TEXT}; font-family: {MONO}; font-size: 14px; font-weight: 800; letter-spacing: 0px; }}
    QLabel#brandMark {{ background: {ACCENT}; color: {ACCENT_INK}; font-family: {MONO}; font-weight: 800;
        font-size: 13px; border-radius: 3px; }}
    /* top stage nav: mono uppercase, numbered like an instrument readout; active = bracketed cyan */
    QLabel#crumb {{ color: {FAINT}; font-family: {MONO}; font-size: 10px; font-weight: 700; padding: 5px 10px;
        border-radius: 3px; letter-spacing: 0px; border: 1px solid transparent; }}
    QLabel#crumbDone {{ color: {MUTED}; font-family: {MONO}; font-size: 10px; font-weight: 700; padding: 5px 10px;
        border-radius: 3px; letter-spacing: 0px; border: 1px solid transparent; }}
    QLabel#crumbOn {{ color: {ACCENT}; font-family: {MONO}; font-size: 10px; font-weight: 800; padding: 5px 10px;
        border-radius: 3px; letter-spacing: 0px; background: {ACCENT2}; border: 1px solid {ACCENT}; }}
    QLabel#crumbSep {{ color: {BORDER2}; font-family: {MONO}; font-size: 10px; padding: 0 3px; }}

    /* ---- left rail (app sections) ---- */
    QWidget#rail {{ background: {RAIL}; border-right: 1px solid {BORDER}; }}
    QToolButton#railBtn {{ background: transparent; color: {FAINT}; border: none; border-left: 2px solid transparent;
        padding: 13px 0 11px 0; font-family: {MONO}; font-size: 10px; font-weight: 700; letter-spacing: 0px; }}
    QToolButton#railBtn:hover {{ color: {TEXT}; background: {SURFACE}; }}
    QToolButton#railBtn:checked {{ color: {ACCENT}; border-left: 2px solid {ACCENT}; background: {SURFACE2}; }}
    QToolButton#dockToggle {{ background: transparent; color: {FAINT}; border: 1px solid {BORDER}; border-radius: 4px;
        font-size: 13px; padding: 1px 6px; }}
    QToolButton#dockToggle:hover {{ color: {ACCENT}; border: 1px solid {ACCENT}; }}

    /* ---- panels + section headers ---- */
    QWidget#buildWorkbench {{ background: {BG}; }}
    QFrame#commandDeck {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
    QLabel#deckTitle {{ color: {TEXT}; font-size: 15px; font-weight: 800; background: transparent; }}
    QLabel#deckMeta {{ color: {FAINT}; font-size: 11px; background: transparent; }}
    QWidget#lanePane {{ background: {RAIL}; border-right: 1px solid {BORDER}; }}
    QWidget#lanePaneInner {{ background: transparent; }}
    QTreeWidget#outlinerTree {{ background: {SURFACE}; border: 1px solid {SOFT}; border-radius: 7px;
        outline: none; padding: 4px; color: {TEXT}; }}
    QTreeWidget#outlinerTree::item {{ padding: 5px 4px; border-radius: 4px; }}
    QTreeWidget#outlinerTree::item:selected {{ background: {ACCENT2}; color: {TEXT}; }}
    QFrame#laneCard, QFrame#laneActive, QFrame#laneDone {{ background: {SURFACE}; border: 1px solid {SOFT};
        border-radius: 7px; }}
    QFrame#laneActive {{ border-left: 3px solid {ACCENT}; background: {ELEV}; }}
    QFrame#laneDone {{ border-left: 3px solid {OK}; }}
    QLabel#laneNum {{ color: {FAINT}; font-family: {MONO}; font-size: 10px; font-weight: 800; background: transparent; }}
    QLabel#laneTitle {{ color: {TEXT}; font-size: 13px; font-weight: 700; background: transparent; }}
    QLabel#laneMeta {{ color: {FAINT}; font-size: 11px; background: transparent; }}
    QFrame#timeline {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
    QFrame#inspectorBanner {{ background: {SURFACE}; border: 1px solid {SOFT}; border-left: 3px solid {ACCENT};
        border-radius: 7px; }}
    QLabel#inspectorName {{ color: {TEXT}; font-size: 15px; font-weight: 800; background: transparent; }}
    QLabel#inspectorKind {{ color: {FAINT}; font-family: {MONO}; font-size: 10px; background: transparent; }}
    QFrame#propGroup {{ background: {SURFACE}; border: 1px solid {SOFT}; border-radius: 7px; }}
    QLabel#propTitle {{ color: {MUTED}; font-family: {MONO}; font-size: 10px; font-weight: 800; letter-spacing: 0px;
        background: transparent; }}
    QLabel#propKey {{ color: {FAINT}; font-size: 11px; background: transparent; }}
    QLabel#propValue {{ color: {TEXT}; font-size: 12px; background: transparent; }}
    QLabel#propMono {{ color: {TEXT}; font-family: {MONO}; font-size: 11px; background: transparent; }}
    QLabel#statusOk {{ color: #04150d; background: {OK}; border-radius: 10px; padding: 2px 9px;
        font-size: 10px; font-weight: 800; }}
    QLabel#statusWarn {{ color: #211403; background: {WARN}; border-radius: 10px; padding: 2px 9px;
        font-size: 10px; font-weight: 800; }}
    QLabel#metricChip {{ color: {TEXT}; background: {SURFACE2}; border: 1px solid {BORDER2}; border-radius: 10px;
        padding: 3px 8px; font-family: {MONO}; font-size: 10px; }}
    QWidget#pane {{ background: {BG}; }}
    QLabel#paneTitle {{ color: {MUTED}; font-family: {MONO}; font-size: 11px; font-weight: 800; letter-spacing: 0px;
        background: transparent; }}
    QFrame#card {{ background: {SURFACE}; border: 1px solid {SOFT}; border-radius: 6px; }}
    QFrame#specCard {{ background: {SURFACE}; border: 1px solid {SOFT}; border-left: 3px solid {BORDER2};
        border-radius: 8px; }}
    QFrame#hr {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
    QLabel#h2 {{ color: {MUTED}; font-family: {MONO}; font-size: 11px; font-weight: 700; letter-spacing: 0px; }}
    QLabel#kpi {{ font-size: 30px; font-weight: 700; font-family: 'JetBrains Mono', Consolas, monospace; }}
    QLabel#kpiLabel {{ color: {FAINT}; font-family: {MONO}; font-size: 10px; font-weight: 700; letter-spacing: 0px; }}
    QLabel#muted {{ color: {MUTED}; }}
    QLabel#faint {{ color: {FAINT}; font-size: 12px; background: transparent; }}
    QLabel#specKey {{ color: {MUTED}; font-size: 12px; }}
    QLabel#specVal {{ color: {TEXT}; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; }}
    QLabel#badgeOk {{ color: {ACCENT_INK}; background: {OK}; border-radius: 9px; padding: 1px 9px;
        font-size: 10px; font-weight: 800; letter-spacing: 0px; }}
    QLabel#badgeWarn {{ color: #2a1c00; background: {WARN}; border-radius: 9px; padding: 1px 9px;
        font-size: 10px; font-weight: 800; }}
    QLabel#badgeBad {{ color: #2a0008; background: {BAD}; border-radius: 9px; padding: 1px 9px;
        font-size: 10px; font-weight: 800; }}

    /* ---- inputs ---- */
    QPlainTextEdit, QComboBox, QLineEdit {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 8px; selection-background-color: {ACCENT}; selection-color: {ACCENT_INK}; }}
    QPlainTextEdit:focus, QComboBox:focus, QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{ background: {SURFACE2}; border: 1px solid {BORDER2};
        selection-background-color: {ELEV}; }}

    /* ---- buttons ---- */
    QPushButton {{ background: {SURFACE2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 5px;
        padding: 8px 12px; }}
    QPushButton:hover {{ border: 1px solid {ACCENT}; background: {ELEV}; }}
    QPushButton:pressed {{ background: {SURFACE}; }}
    QPushButton:disabled {{ color: {FAINT}; border-color: {BORDER}; background: {SURFACE}; }}
    QPushButton#primary {{ background: {ACCENT}; color: {ACCENT_INK}; border: none; font-weight: 700; padding: 9px 15px; }}
    QPushButton#primary:hover {{ background: #ee8a34; }}
    QPushButton#primary:pressed {{ background: #b85a1b; }}
    QPushButton#primary:disabled {{ background: {SURFACE2}; color: {FAINT}; }}
    QPushButton#ghost {{ background: transparent; border: 1px solid {BORDER}; color: {MUTED}; }}
    QPushButton#ghost:hover {{ color: {TEXT}; border: 1px solid {ACCENT}; }}
    QPushButton#chip {{ background: transparent; border: 1px dashed {BORDER}; color: {MUTED}; padding: 6px 10px;
        border-radius: 8px; font-size: 12px; text-align: left; }}
    QPushButton#chip:hover {{ border: 1px solid {ACCENT}; color: {TEXT}; background: {SURFACE}; }}

    /* ---- inspector tabs ---- */
    QTabWidget::pane {{ border: none; top: -1px; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{ background: transparent; color: {FAINT}; padding: 7px 10px; margin-right: 1px;
        border: none; border-bottom: 2px solid transparent; font-size: 11px; font-weight: 700; }}
    QTabBar::tab:hover {{ color: {TEXT}; }}
    QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}

    /* ---- entity tree ---- */
    QTreeWidget, QListWidget {{ background: {SURFACE}; border: 1px solid {SOFT}; border-radius: 8px;
        outline: none; padding: 4px; }}
    QTreeWidget::item, QListWidget::item {{ padding: 5px 4px; border-radius: 5px; color: {TEXT}; }}
    QTreeWidget::item:hover, QListWidget::item:hover {{ background: {SURFACE2}; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{ background: {ELEV}; color: {ACCENT}; }}
    QHeaderView::section {{ background: {SURFACE2}; color: {MUTED}; border: none; padding: 5px;
        font-size: 10px; font-weight: 700; letter-spacing: 0px; }}

    /* ---- console tray + status ---- */
    QWidget#tray {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
    QPlainTextEdit#console {{ background: {RAIL}; border: 1px solid {BORDER}; border-radius: 8px;
        font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; }}
    QStatusBar {{ background: {SURFACE}; color: {TEXT}; border-top: 1px solid {BORDER}; font-size: 12px; }}
    QStatusBar QLabel {{ color: {MUTED}; }}
    QStatusBar::item {{ border: none; }}

    /* ---- scroll + slider + progress ---- */
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {BORDER2}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {BORDER2}; border-radius: 5px; min-width: 30px; }}
    QSlider::groove:horizontal {{ height: 5px; background: {SURFACE2}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
    QSlider::handle:horizontal:disabled {{ background: {BORDER2}; }}
    QSlider::sub-page:horizontal:disabled {{ background: {SURFACE2}; }}
    QProgressBar {{ background: {SURFACE2}; border: none; border-radius: 4px; height: 6px; text-align: center; }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
    QToolTip {{ background: {ELEV}; color: {TEXT}; border: 1px solid {BORDER2}; padding: 4px 7px; }}

    /* ---- assistant chat ---- */
    QWidget#chatpane {{ background: {BG}; border-right: 1px solid {BORDER}; }}
    QScrollArea#chatScroll {{ background: {BG}; }}
    QFrame#bubbleUser {{ background: {ELEV}; border: 1px solid {BORDER}; border-radius: 12px; }}
    QLabel#bubbleText {{ color: {TEXT}; font-size: 13px; background: transparent; }}
    QLabel#aiAvatar {{ background: {ACCENT}; color: #ffffff; font-weight: 800; font-size: 11px;
        border-radius: 5px; padding: 1px 5px; }}
    QLabel#aiName {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0px; }}
    QLabel#aiText {{ color: {TEXT}; font-size: 13px; background: transparent; }}
    QLabel#sysLine {{ color: {FAINT}; font-size: 12px; background: transparent; }}
    QFrame#welcome {{ background: {RAIL}; border: 1px solid {BORDER}; border-left: 3px solid {BORDER2}; border-radius: 7px; }}
    QLabel#welcomeTitle {{ color: {TEXT}; font-size: 13px; font-weight: 700; background: transparent; }}
    QPushButton#suggest {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 10px;
        color: {TEXT}; padding: 12px 14px; text-align: left; font-size: 13px; font-weight: 600; }}
    QPushButton#suggest:hover {{ border: 1px solid {ACCENT}; background: {ELEV}; color: {TEXT}; }}
    QWidget#composer {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
    QPlainTextEdit#composerInput {{ background: {SURFACE2}; border: 1px solid {BORDER2}; border-radius: 10px;
        padding: 9px 11px; font-size: 13px; }}
    QPlainTextEdit#composerInput:focus {{ border: 1px solid {ACCENT}; }}
    QWidget#quickbar {{ background: {SURFACE}; }}
    """


def _chip_qss(color: str = MUTED) -> str:
    """Shared pill/chip stylesheet  -  used by status chips, the project pill and the outcome badge."""
    return (f"color: {color}; background: {SURFACE2}; border: 1px solid {BORDER};"
            f"border-radius: 11px; padding: 3px 10px; font-size: 12px;")


def _chip(text: str, color: str = MUTED) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(_chip_qss(color))
    return lab


class _Composer(QPlainTextEdit):
    """Chat composer: Enter sends, Shift+Enter inserts a newline (the Claude/chat convention)."""

    def __init__(self, on_submit):
        super().__init__()
        self._on_submit = on_submit

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
            self._on_submit(); return
        super().keyPressEvent(e)


# ============================================================================ workers
class JobWorker(QThread):
    """Runs a build/evaluate/iterate job off the UI thread; streams progress via signals."""
    progress = Signal(dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, workspace: Path, project_dir: Path, agent, message: str):
        super().__init__()
        self.workspace, self.project_dir, self.agent, self.message = workspace, project_dir, agent, message

    def run(self):
        try:
            from virturoid.services.agent import parse_intent
            intent, params = parse_intent(self.message)

            def emit(ev: dict):
                self.progress.emit(ev)

            if intent in {"build", "iterate", "adjust_and_rebuild"}:
                from virturoid.services.autonomous_build import autonomous_build
                if intent == "build":
                    self.agent.prompt = params.get("prompt", self.message)
                elif intent == "adjust_and_rebuild" and params.get("reach_m"):
                    self.agent.reach_m = params["reach_m"]
                    self.agent.prompt = f"{self.agent.prompt} (reach {self.agent.reach_m} m)"
                elif intent == "iterate":
                    self.agent.target_success = min(1.0, self.agent.target_success + 0.1)
                if not self.agent.prompt:
                    self.failed.emit("Describe a robot to build first."); return
                report = autonomous_build(self.agent.prompt, self.project_dir,
                                          target_success_rate=self.agent.target_success,
                                          memory_dir=self.workspace / "memory", progress=emit)
                self.finished_ok.emit({"intent": intent,
                                       "message": f"Build {'succeeded' if report.succeeded else 'did not reach target'}: "
                                                  f"task success {report.final_success_rate:.0%}."})
            else:
                emit({"stage": intent, "message": f"Working on: {self.message}"})
                resp = self.agent.handle(self.message)
                self.finished_ok.emit({"intent": intent, "message": resp.message})
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ComposeWorker(QThread):
    """Compose a robot from a prompt and hand back its MJCF (off the UI thread). Fast  -  no physics
    run  -  so the viewport shows the actual composed body almost instantly."""
    ready = Signal(dict, str)   # (robot_summary, model_xml)
    failed = Signal(str)

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            from virturoid.services.robot_factory import build_robot
            res = build_robot(self.prompt)          # (A) real production model if it matches, else (B) procedural
            if res["kind"] == "real":
                summary = {
                    "species": res["label"], "name": res["label"],
                    "robot_class": f"real  /  {res['robot_kind']}", "dof": res["actuated"],
                    "links": [], "end_effectors": [], "valid": True, "design_source": "real",
                    "_real": res, "_prompt": self.prompt,
                }
                self.ready.emit(summary, res["mjcf"])
                return
            from virturoid.services.gene_compiler import gene_to_meshed_mjcf
            gene = res["gene"]
            # High-fidelity meshed viewport (slim links + collars + joint housings); runs on this worker
            # thread so the mesh-gen cost stays off the UI, is cached, and falls back to primitives if the
            # CAD kernel is unavailable. Physics is identical to the primitive model the task layer runs on.
            xml = gene_to_meshed_mjcf(gene)
            ee = gene.end_effector_type
            summary = {
                "species": gene.species, "name": gene.species or gene.id,
                "robot_class": gene.robot_class, "dof": len(gene.actuated_joints()),
                "links": [s.name for s in gene.segments],
                "end_effectors": [] if (not ee or ee == "none") else [ee],
                "valid": not gene.validate(),
                "design_source": getattr(gene, "design_source", "heuristic"),
                "_gene": gene, "_prompt": self.prompt,    # carried in-process so a task can run on it
            }
            self.ready.emit(summary, xml)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TaskWorker(QThread):
    """Run the GENERAL task layer on a composed robot (off the UI thread): propose a task from the prompt,
    verify it against the robot's morphology + skills, and run it  -  returning honest feasibility + score."""
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, prompt: str, gene):
        super().__init__()
        self.prompt, self.gene = prompt, gene

    def run(self):
        try:
            from virturoid.services.task_executor import evaluate_task
            self.done.emit(evaluate_task(self.prompt, self.gene, record=True))   # capture a replay for the viewport
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class EpisodeWorker(QThread):
    """Run the built robot's episode (recording per-frame geom poses) off the UI thread and return the
    scene MJCF + poses  -  the interactive viewport then renders them live with whatever camera the user
    is flying, instead of a pre-baked video."""
    ready = Signal(dict, str)   # (view_dict, model_xml)
    failed = Signal(str)

    def __init__(self, project_dir: Path, scene_index: int = 0):
        super().__init__()
        self.project_dir, self.scene_index = project_dir, scene_index

    def run(self):
        try:
            from virturoid.services.viewer_sim import simulate_episode_for_viewer
            view = simulate_episode_for_viewer(self.project_dir, scene_index=self.scene_index)
            compiled = json.loads((self.project_dir / "simulation" / "mujoco" /
                                   "compiled_scene_index.json").read_text(encoding="utf-8"))
            by_scene = {e["scene_id"]: e["mujoco_xml"] for e in compiled.get("scenes", [])}
            xml = (self.project_dir / by_scene[view["scene_id"]]).read_text(encoding="utf-8")
            self.ready.emit(view, xml)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class LearnWorker(QThread):
    """Learn-on-request: train a locomotion policy for THIS body (off the UI thread), replay it, bank the
    skill, and commit species-specific tips to memory so the next build of the species reuses it."""
    note = Signal(str)
    done = Signal(dict, dict, str)   # (result, replay_view, model_xml)
    failed = Signal(str)

    def __init__(self, gene, workspace: Path, warm_start=None, use_gpu=False, gpu_iters=80, species=None):
        super().__init__()
        self.gene, self.workspace, self.warm_start = gene, workspace, warm_start
        self.use_gpu, self.gpu_iters, self.species = use_gpu, gpu_iters, species

    def _train_gpu(self):
        import mujoco
        from virturoid.services.gpu_trainer import train_mjcf_on_gpu
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import MorphPolicy, robot_mjcf
        from virturoid.services.morph_trainer import forward_score
        imported = isinstance(self.gene, str)
        species = (self.species or getattr(self.gene, "species", None)
                   or getattr(self.gene, "robot_class", None) or "robot")
        fd = encode_robot(mujoco.MjModel.from_xml_string(robot_mjcf(self.gene))).feature_dim
        base = forward_score(self.gene, MorphPolicy(fd, seed=0), steps=170)
        out = str(Path("models") / ("gpu_" + species.replace("/", "_") + ".npz"))
        if imported:
            npz = train_mjcf_on_gpu(self.gene, out_path=out, iters=self.gpu_iters, cpg=True,
                                    progress=lambda m: self.note.emit(m))
            critic_rounds = 0
        else:
            # AI-ASSISTED gait loop (the real MVP path): trot-CPG prior so it can't collapse from a standstill +
            # the LLM reward critic redesigns the reward across rounds when a backend is on (offline -> one CPG run).
            from virturoid.services.assisted_trainer import ai_critic_gait_loop
            from virturoid.services.llm_client import get_llm
            llm = get_llm("designer")
            res = ai_critic_gait_loop(self.gene, llm=llm, rounds=(3 if llm is not None else 1), iters=35,
                                      models_dir="models", progress=lambda m: self.note.emit(m))
            npz = res.get("npz"); critic_rounds = res.get("rounds", 0)
        if not npz:
            return None
        pol = MorphPolicy.from_npz(npz)
        return {"policy": pol, "npz_path": npz, "score": float(forward_score(self.gene, pol, steps=200)),
                "baseline": float(base), "warm_started": False, "species": species, "backend": "gpu",
                "critic_rounds": critic_rounds}

    def run(self):
        try:
            from virturoid.services.learn_locomotion import learn_locomotion, rollout_view
            res = None
            if self.use_gpu:
                res = self._train_gpu()
                if res is None:
                    self.note.emit("GPU training unavailable  -  falling back to on-device training...")
            if res is None:
                # recipe=True is the anti-collapse control (PD-to-default + obs-norm + terminate + velocity-track)
                # that actually WALKS; the old (recipe=False) torque-residual path collapses to a fall-forward flop.
                # The normalizer is banked WITH the policy, and rollout_view auto-detects it for a faithful replay.
                # adaptive="auto" inertia-scales the per-joint gains for bodies far from the reference quad (a
                # humanoid/heavy build), so the SAME recipe walks any morphology; a standard quad keeps scalar gains.
                res = learn_locomotion(self.gene, generations=14, pop=18, steps=420, seeds=2, recipe=True,
                                       adaptive="auto", warm_start=self.warm_start, species=self.species,
                                       progress=lambda m: self.note.emit(m))
            self.note.emit("recording the learned motion...")
            view, xml = rollout_view(self.gene, res["policy"], steps=340)
            try:                                  # bank the skill + commit species tips (the flywheel)
                from virturoid.services.memory_db import MemoryDB
                from virturoid.services.policy_flywheel import bank_morph_policy
                with MemoryDB(self.workspace / "memory" / "virturoid_memory.db") as db:
                    species = res["species"]
                    db.upsert_species_node(species, robot_class=getattr(self.gene, "robot_class", None))
                    bank = bank_morph_policy(res["npz_path"], self.gene, db, task_type="locomotion")
                    _fm = res.get("forward_m")
                    _fwd = f", {_fm:.2f} m forward" if isinstance(_fm, (int, float)) else ""
                    tip = (f"Learned a locomotion policy (gait reward {res['score']:.2f} vs "
                           f"{res['baseline']:.2f} untrained{_fwd}); banked as a reusable skill  -  warm-start the "
                           f"next {species} from it instead of training from scratch.")
                    db.add_species_tip(species, tip, audience="trainer")
                    res["skill_id"] = bank.get("skill_id"); res["tip"] = tip
            except Exception as exc:  # noqa: BLE001
                res["bank_error"] = str(exc)
            self.done.emit(res, view, xml)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class GpuPingWorker(QThread):
    """Check GPU-box reachability once at startup (off the UI thread) so GPU training can be the default."""
    result = Signal(bool)

    def run(self):
        try:
            from virturoid.services.gpu_trainer import gpu_available
            self.result.emit(gpu_available(timeout=10))
        except Exception:  # noqa: BLE001
            self.result.emit(False)


# ============================================================================ 3D viewport
def _inject_framebuffer(xml: str, w: int, h: int) -> str:
    """Ensure the model declares an offscreen framebuffer big enough to render at (w, h)  -  MuJoCo's
    default offscreen buffer is only 640x480, which would cap viewport quality."""
    if "offwidth" in xml:
        return xml
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    if "<visual/>" in xml:
        return xml.replace("<visual/>", "<visual>" + g + "</visual>", 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


class MujocoViewport(QWidget):
    """A live, interactive MuJoCo 3D viewport  -  orbit (left-drag), pan (right/shift-drag) and zoom
    (wheel) around the actual model, the Blender/RViz pattern. Renders MuJoCo offscreen to a QImage on
    demand (driven by mouse + playback), so it is a real 3D view you fly around, not a pre-baked video.
    It shows either a static composed model (Preview) or a recorded episode you can orbit while it plays.
    """
    RW, RH = 1024, 720    # offscreen render resolution (scaled to fit the widget)

    def __init__(self):
        super().__init__()
        import numpy as np
        import mujoco
        self._mj, self._np = mujoco, np
        self.model = self.data = self.renderer = None
        self.cam = mujoco.MjvCamera()
        self._mat = np.zeros(9)
        self._pix = None
        self.frames: list = []
        self.idx = 0
        self._drag = None
        self._render_pending = False
        self._scene_cb = None
        self._reset_cam()

        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        self.canvas = QLabel("Viewport ready\n\nDescribe a robot in the design brief\nand watch it build here in real physics\n\n"
                             "left-drag orbit / right-drag pan / scroll zoom")
        self.canvas.setAlignment(Qt.AlignCenter)
        # flat dark stage (standard for a 3D viewport); a loaded robot replaces this with the rendered pixmap.
        self.canvas.setStyleSheet(f"QLabel {{ background: #0a0c12; color: {FAINT}; font-size: 14px; }}")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(360, 260)
        self.canvas.setMouseTracking(True)
        v.addWidget(self.canvas, 1)

        bar = QHBoxLayout(); bar.setContentsMargins(10, 8, 10, 8); bar.setSpacing(8)
        self.play_btn = QPushButton("Play"); self.play_btn.setEnabled(False); self.play_btn.setFixedWidth(72)
        self.play_btn.clicked.connect(self.toggle_play)
        self.scene_box = QComboBox(); self.scene_box.setEnabled(False); self.scene_box.setFixedWidth(132)
        self.scene_box.addItem("static preview")
        self.scene_box.currentIndexChanged.connect(self._scene_changed)
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._scrub)
        self.frame_lab = QLabel(" - "); self.frame_lab.setObjectName("faint")
        self.reset_btn = QPushButton("Reset view"); self.reset_btn.setToolTip("Recenter the camera")
        self.reset_btn.setMinimumWidth(96); self.reset_btn.clicked.connect(self.reset_view)
        self.outcome = _chip(" - "); self.outcome.setMinimumWidth(120)
        bar.addWidget(self.play_btn); bar.addWidget(self.scene_box)
        bar.addWidget(self.slider, 1); bar.addWidget(self.frame_lab)
        bar.addWidget(self.reset_btn); bar.addWidget(self.outcome)
        bar_w = QWidget(); bar_w.setStyleSheet(f"background: {SURFACE}; border-top: 1px solid {BORDER};")
        bar_w.setLayout(bar); v.addWidget(bar_w)

        self.timer = QTimer(self); self.timer.setInterval(55); self.timer.timeout.connect(self._tick)

    # ---- public API ----
    def set_scene_callback(self, cb):
        self._scene_cb = cb

    def has_live_episode(self) -> bool:
        """True only when a real recorded episode (>=1 frame) is loaded in the viewport. The Evidence/score
        panels gate on this so the app can NEVER show a task score over a frozen/empty viewport."""
        return bool(getattr(self, "_has_episode", False))

    def show_message(self, text: str):
        self.timer.stop(); self.play_btn.setText("Play")
        self._pix = None
        self._has_episode = False
        self.canvas.setText(text)

    def _pose_natural(self):
        """Bend the joints into a gentle articulated pose so multi-segment limbs read as limbs, not a pole."""
        if self.model is None or self.data is None:
            return
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(self._mj.mjtJoint.mjJNT_HINGE):
                ang = 0.45 * (1 if j % 2 else -1)
                if self.model.jnt_limited[j]:
                    lo, hi = float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])
                    ang = max(lo + 0.05, min(hi - 0.05, ang))
                self.data.qpos[int(self.model.jnt_qposadr[j])] = ang
        self._mj.mj_forward(self.model, self.data)

    def load_static(self, model_xml: str, summary: dict | None = None):
        """Show a composed model the user can orbit (no episode)."""
        if not self._build_model(model_xml):
            return
        self._pose_natural()
        self.frames = []; self.idx = 0
        self._has_episode = False               # a static preview is NOT a simulated episode
        self.play_btn.setEnabled(False); self.slider.setEnabled(False)
        self.scene_box.setEnabled(False); self.scene_box.clear(); self.scene_box.addItem("static preview")
        self.frame_lab.setText("model")
        cls = (summary or {}).get("robot_class", "robot")
        dof = (summary or {}).get("dof", 0)
        self._set_outcome(f"preview  /  {cls}  /  {dof} DOF", MUTED)
        self.reset_view()

    def load_episode(self, view: dict, model_xml: str):
        """Show a recorded episode's per-frame poses, orbitable while it plays."""
        if not self._build_model(model_xml):
            return
        self.frames = view.get("frames", []) or []
        self.idx = 0
        self.set_scenes(view.get("scenes", []), view.get("scene_index", 0))
        oc = view.get("outcome", {})
        col = OK if oc.get("status") == "success" else WARN
        self._set_outcome(f"{oc.get('status', ' - ')}  /  {oc.get('placed_count', 0)}/{oc.get('block_count', 0)}", col)
        has = len(self.frames) > 0
        self._has_episode = has                 # a real simulated episode is loaded only when there are frames
        self.play_btn.setEnabled(has); self.slider.setEnabled(has)
        self.slider.setMaximum(max(0, len(self.frames) - 1)); self.slider.setValue(0)
        if has:
            self._apply_frame(0); self._render(); self.toggle_play()
        else:
            self.frame_lab.setText("model"); self._render()

    def set_scenes(self, scenes: list, current: int):
        self.scene_box.blockSignals(True); self.scene_box.clear()
        for s in scenes:
            self.scene_box.addItem(f"Scene {s['index'] + 1}: {s.get('name', s['id'])}", s["index"])
        if scenes:
            self.scene_box.setCurrentIndex(current)
        else:
            self.scene_box.addItem("static preview")
        self.scene_box.setEnabled(len(scenes) > 1)
        self.scene_box.blockSignals(False)

    def body_tree(self) -> list:
        """[(body_name, joint_count), ...] for the loaded model  -  feeds the STRUCTURE panel for
        real / imported robots whose link names aren't known from a composed gene."""
        if self.model is None:
            return []
        out = []
        for b in range(1, self.model.nbody):       # skip the world body (index 0)
            name = self.model.body(b).name or f"body{b}"
            out.append((name, int(self.model.body_jntnum[b])))
        return out

    # ---- model + render ----
    def _build_model(self, model_xml: str) -> bool:
        try:
            self.model = self._mj.MjModel.from_xml_string(_inject_framebuffer(model_xml, self.RW, self.RH))
            # Display-only: lift near-black Menagerie materials + neutral lighting so REAL production models
            # (Unitree H1/Go1, etc.) render bright and legible in the viewport instead of as black silhouettes.
            try:
                from virturoid.services.real_model_library import normalize_display
                normalize_display(self.model)
            except Exception:  # noqa: BLE001 - rendering must never fail on a cosmetic step
                pass
            self.data = self._mj.MjData(self.model)
            self._mj.mj_forward(self.model, self.data)
            if self.renderer is not None:
                try: self.renderer.close()
                except Exception: pass  # noqa: BLE001
            self.renderer = self._mj.Renderer(self.model, height=self.RH, width=self.RW)
            return True
        except Exception as exc:  # noqa: BLE001
            self.show_message("3D render unavailable on this machine:\n" + str(exc) +
                              "\n(Build, results, and metrics still work.)")
            self.model = self.renderer = None
            return False

    def _render(self):
        if self.renderer is None or self.data is None:
            return
        try:
            self.renderer.update_scene(self.data, self.cam)
            img = self.renderer.render()
        except Exception:  # noqa: BLE001
            return
        h, w, _ = img.shape
        qimg = QImage(img.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(qimg)
        self._blit()

    def _blit(self):
        if self._pix is not None and not self._pix.isNull():
            self.canvas.setPixmap(self._pix.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _request_render(self):
        """Coalesce renders from rapid mouse-move / wheel events into one per event-loop pass  -  the
        camera math still runs every event, but the offscreen render happens at most once per tick."""
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._do_render)

    def _do_render(self):
        self._render_pending = False
        self._render()

    def _apply_frame(self, i: int):
        if not (0 <= i < len(self.frames)) or self.data is None:
            return
        self.idx = i
        for g, pose in enumerate(self.frames[i]):
            self.data.geom_xpos[g] = pose[:3]
            self._mj.mju_quat2Mat(self._mat, self._np.asarray(pose[3:7], dtype=float))
            self.data.geom_xmat[g] = self._mat
        self.frame_lab.setText(f"{i + 1} / {len(self.frames)}")
        self.slider.blockSignals(True); self.slider.setValue(i); self.slider.blockSignals(False)

    # ---- playback ----
    def _tick(self):
        if not self.frames:
            self.toggle_play(force_stop=True); return
        nxt = self.idx + 1
        if nxt >= len(self.frames):
            self.toggle_play(force_stop=True); return
        self._apply_frame(nxt); self._render()

    def toggle_play(self, force_stop: bool = False):
        if self.timer.isActive() or force_stop:
            self.timer.stop(); self.play_btn.setText("Play")
        elif self.frames:
            if self.idx >= len(self.frames) - 1:
                self._apply_frame(0)
            self.timer.start(); self.play_btn.setText("Pause")

    def _scrub(self, v: int):
        if self.timer.isActive():
            self.toggle_play(force_stop=True)
        self._apply_frame(v); self._render()

    def _scene_changed(self, _i: int):
        if self._scene_cb and self.scene_box.isEnabled():
            self._scene_cb(self.scene_box.currentData())

    # ---- camera ----
    def _reset_cam(self):
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 135.0, -18.0, 1.7
        self.cam.lookat[:] = [0.3, 0.0, 0.35]

    def reset_view(self):
        self._reset_cam(); self._render()

    def _set_outcome(self, text: str, color: str):
        self.outcome.setText(text)
        self.outcome.setStyleSheet(_chip_qss(color))

    def mousePressEvent(self, e):
        self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is None or self.model is None:
            return
        import math
        p = e.position(); dx = p.x() - self._drag.x(); dy = p.y() - self._drag.y(); self._drag = p
        pan = (e.buttons() & Qt.RightButton) or ((e.buttons() & Qt.LeftButton) and (e.modifiers() & Qt.ShiftModifier))
        if pan:
            az, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
            right = (-math.sin(az), math.cos(az), 0.0)
            up = (-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el))
            s = 0.0022 * self.cam.distance
            for i in range(3):
                self.cam.lookat[i] += (-dx * right[i] + dy * up[i]) * s
        elif e.buttons() & Qt.LeftButton:
            self.cam.azimuth = (self.cam.azimuth - dx * 0.4) % 360
            self.cam.elevation = max(-89.0, min(89.0, self.cam.elevation - dy * 0.4))
        else:
            return
        self._request_render()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def wheelEvent(self, e):
        if self.model is None:
            return
        self.cam.distance *= 0.86 if e.angleDelta().y() > 0 else 1.16
        self.cam.distance = max(0.2, min(10.0, self.cam.distance))
        self._request_render()

    def resizeEvent(self, e):
        self._blit()
        super().resizeEvent(e)


# ============================================================================ main window
class MainWindow(QMainWindow):
    """Tri-pane robot studio: command bar + stage breadcrumb on top, a slim section rail on the left,
    and the Intent/Structure -> Viewport -> Inspector tri-pane (the report's central recommendation),
    over a collapsible console tray. Native desktop, not a web view."""

    STAGES = ["Design", "Simulate", "Train", "Evaluate", "Export"]

    def __init__(self, workspace: Path):
        super().__init__()
        from virturoid.services.agent import VirturoidAgent
        self.workspace = Path(workspace)
        self.agent = VirturoidAgent(self.workspace, target_success=0.8)
        self.project_dir = self.agent.project_dir
        self.built = False
        self.worker = self.render_worker = self.compose_worker = self.task_worker = self.learn_worker = None
        self.composed_gene = None         # last composed robot (a RobotGene) OR an imported MJCF string
        self.composed_meta = None         # for imported models: {name, parts, actuated, free_base, ...}
        self.composed_prompt = ""
        self._last_policy = None          # last learned policy for THIS body -> warm-start to keep improving
        self.stage = "Design"
        self._artifact_gate_states = None
        self._gpu_ok = False              # set by the startup reachability check; GPU is the default when True
        self._auto = False                # a fresh request auto-runs the pipeline (compose -> train/run)

        self.setWindowTitle("Virturoid Studio")
        self.resize(1460, 920)
        # Min size must exceed the sum of child min widths (rail 78 + left 270 + viewport ~360 + inspector 420
        # + splitter handles) or the right Inspector clips off-screen when the window isn't maximized.
        self.setMinimumSize(1340, 780)

        self.viewport = MujocoViewport()
        self.viewport.set_scene_callback(self._load_episode)

        self._build_menu()
        self._build_ui()
        self._build_statusbar()
        self._set_stage("Design")
        self.refresh_project()
        self._gpu_ping = GpuPingWorker(); self._gpu_ping.result.connect(self._on_gpu_ping); self._gpu_ping.start()

    # ============================================================ chrome
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        new_action = QAction("New Design", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._on_new)
        file_menu.addAction(new_action)
        import_action = QAction("Import Robot Model...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self._import_model)
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        run_menu = self.menuBar().addMenu("&Run")
        compose_action = QAction("Build / Train Current Design", self)
        compose_action.setShortcut(QKeySequence("Ctrl+B"))
        compose_action.triggered.connect(self._full_build)
        run_menu.addAction(compose_action)
        task_action = QAction("Run Task", self)
        task_action.setShortcut(QKeySequence("Ctrl+R"))
        task_action.triggered.connect(self._start_task)
        run_menu.addAction(task_action)
        replay_action = QAction("Replay Episode", self)
        replay_action.setShortcut(QKeySequence("Ctrl+E"))
        replay_action.triggered.connect(lambda: self._load_episode(0))
        run_menu.addAction(replay_action)

        view_menu = self.menuBar().addMenu("&View")
        build_action = QAction("Build Workbench", self)
        build_action.triggered.connect(lambda: self._go_section(0))
        memory_action = QAction("Memory Library", self)
        memory_action.triggered.connect(lambda: self._go_section(1))
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(lambda: self._go_section(2))
        view_menu.addActions([build_action, memory_action, settings_action])

    def _build_ui(self):
        root = QWidget(); col = QVBoxLayout(root); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(0)
        col.addWidget(self._top_bar())
        body = QWidget(); brow = QHBoxLayout(body); brow.setContentsMargins(0, 0, 0, 0); brow.setSpacing(0)
        brow.addWidget(self._rail())
        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(self._build_page())      # 0  -  authoring
        self.main_stack.addWidget(self._library_page())    # 1  -  flywheel
        self.main_stack.addWidget(self._settings_page())   # 2  -  settings
        brow.addWidget(self.main_stack, 1)
        col.addWidget(body, 1)
        self.setCentralWidget(root)

    def _top_bar(self) -> QWidget:
        bar = QWidget(); bar.setObjectName("topbar"); bar.setFixedHeight(52)
        h = QHBoxLayout(bar); h.setContentsMargins(14, 0, 14, 0); h.setSpacing(10)
        mark = QLabel("V"); mark.setObjectName("brandMark"); mark.setFixedSize(24, 24); mark.setAlignment(Qt.AlignCenter)
        name = QLabel("VIRTUROID STUDIO"); name.setObjectName("brand")
        sub = QLabel("native robot authoring workstation"); sub.setObjectName("deckMeta")
        h.addWidget(mark); h.addWidget(name); h.addWidget(sub)
        h.addStretch(1)
        self.proj_pill = _chip("No robot yet")
        self.proj_pill.setMaximumWidth(240)        # never let a long species name push the buttons off-screen
        h.addWidget(self.proj_pill)
        impb = QPushButton("Import Model"); impb.setObjectName("ghost")
        impb.setToolTip("Import an existing robot model (MJCF / URDF) to iterate on")
        impb.clicked.connect(self._import_model)
        newb = QPushButton("New Design"); newb.setObjectName("ghost"); newb.clicked.connect(self._on_new)
        setb = QPushButton("Settings"); setb.setObjectName("ghost"); setb.clicked.connect(lambda: self._go_section(2))
        h.addWidget(impb); h.addWidget(newb); h.addWidget(setb)
        return bar

    def _crumb_widget(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(2)
        self.crumb_labels = {}
        for i, s in enumerate(self.STAGES):
            if i:
                sep = QLabel(">"); sep.setObjectName("crumbSep"); h.addWidget(sep)
            lab = QLabel(s.upper()); lab.setObjectName("crumb")
            # DPI-proof: keep each label at its natural (sizeHint) width so the layout never truncates it
            lab.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            self.crumb_labels[s] = lab; h.addWidget(lab)
        return w

    def _set_stage(self, name: str):
        if name not in self.crumb_labels:
            return
        self.stage = name; idx = self.STAGES.index(name)
        for i, s in enumerate(self.STAGES):
            lab = self.crumb_labels[s]
            lab.setObjectName("crumbOn" if i == idx else ("crumbDone" if i < idx else "crumb"))
            lab.style().unpolish(lab); lab.style().polish(lab)
        if hasattr(self, "lane_cards"):
            for s, cards in self.lane_cards.items():
                lane_idx = self.STAGES.index(s)
                state = "laneActive" if lane_idx == idx else ("laneDone" if lane_idx < idx else "laneCard")
                for card in cards:
                    card.setObjectName(state)
                    card.style().unpolish(card); card.style().polish(card)
                    status = getattr(self, "lane_status_labels", {}).get(card)
                    if status is not None:
                        status.setText("active" if lane_idx == idx else ("done" if lane_idx < idx else "waiting"))
            if getattr(self, "_artifact_gate_states", None):
                self._apply_gate_states(self._artifact_gate_states)

    def _rail(self) -> QWidget:
        w = QWidget(); w.setObjectName("rail"); w.setFixedWidth(78)
        v = QVBoxLayout(w); v.setContentsMargins(0, 8, 0, 8); v.setSpacing(2)
        self.rail_group = QButtonGroup(self); self.rail_group.setExclusive(True)
        for label, page in [("BUILD", 0), ("MEMORY", 1), ("SETTINGS", 2)]:
            b = QToolButton(); b.setObjectName("railBtn"); b.setText(label); b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly); b.setFixedWidth(78)
            b.clicked.connect(lambda _=False, p=page: self._go_section(p))
            self.rail_group.addButton(b, page); v.addWidget(b)
        self.rail_group.button(0).setChecked(True)
        v.addStretch(1)
        return w

    def _go_section(self, page: int):
        # Library (1) + Settings (2) are read-only snapshots  -  rebuild on entry so the flywheel counts
        # and GPU status reflect the latest state without restarting the app. Build (0) keeps its state.
        if page == 1:
            self._reload_stack_page(1, self._library_page())
        elif page == 2:
            self._reload_stack_page(2, self._settings_page())
        self.main_stack.setCurrentIndex(page)
        btn = self.rail_group.button(page)
        if btn and not btn.isChecked():
            btn.setChecked(True)

    def _reload_stack_page(self, index: int, widget: QWidget):
        old = self.main_stack.widget(index)
        self.main_stack.insertWidget(index, widget)   # new widget takes `index`; `old` shifts to index+1
        if old is not None:
            self.main_stack.removeWidget(old)
            old.deleteLater()

    # ============================================================ build page (desktop workbench)
    def _build_page(self) -> QWidget:
        page = QWidget(); page.setObjectName("buildWorkbench")
        v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self._command_deck())

        self._split = QSplitter(Qt.Horizontal)
        self._dock_left = self._left_pane()          # AI assistant  -  collapsible
        self._dock_right = self._right_pane()         # inspector  -  collapsible
        self._split.addWidget(self._dock_left)
        self._split.addWidget(self._center_pane())    # viewport  -  the hero, never collapses
        self._split.addWidget(self._dock_right)
        self._split.setStretchFactor(0, 0); self._split.setStretchFactor(1, 1); self._split.setStretchFactor(2, 0)
        self._split.setCollapsible(1, False); self._split.setSizes([310, 790, 440])
        v.addWidget(self._split, 1)
        v.addWidget(self._timeline_pane(), 0)
        return page

    def _command_deck(self) -> QWidget:
        deck = QFrame(); deck.setObjectName("commandDeck"); deck.setFixedHeight(126)
        v = QVBoxLayout(deck); v.setContentsMargins(16, 8, 16, 8); v.setSpacing(6)

        top = QHBoxLayout(); top.setSpacing(12)
        title_col = QVBoxLayout(); title_col.setSpacing(1)
        title = QLabel("Mission Command"); title.setObjectName("deckTitle")
        meta = QLabel("Prompt -> morphology -> sim scene -> task run -> package evidence")
        meta.setObjectName("deckMeta")
        title_col.addWidget(title); title_col.addWidget(meta)
        top.addLayout(title_col); top.addSpacing(14); top.addWidget(self._crumb_widget()); top.addStretch(1)
        # "Build + Train" is the honest label: this runs the full build+train pipeline (it does NOT just package).
        self.build_pkg_btn = QPushButton("Build + Train"); self.build_pkg_btn.setObjectName("ghost")
        self.build_pkg_btn.setToolTip("Design, build, and train this robot into a full package on disk")
        self.build_pkg_btn.clicked.connect(self._full_build)
        top.addWidget(self.build_pkg_btn)
        # Give the EXPORT stage a real deliverable: reveal the built package folder (Robot Genome, CAD, reports,
        # ROS2 bundle) on disk so a reviewer can open the actual artifacts, not just a crumb that lights up.
        self.open_pkg_btn = QPushButton("Open package"); self.open_pkg_btn.setObjectName("ghost")
        self.open_pkg_btn.setToolTip("Reveal the built package folder (Robot Genome, CAD, reports, ROS2) on disk")
        self.open_pkg_btn.setEnabled(False)
        self.open_pkg_btn.clicked.connect(self._open_package)
        top.addWidget(self.open_pkg_btn)
        v.addLayout(top)

        row = QHBoxLayout(); row.setSpacing(10)
        self.composer = _Composer(self._chat_send); self.composer.setObjectName("composerInput")
        self.composer.setPlaceholderText("Describe the robot, task, constraints, parts, sensors, or scene.")
        self.composer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.composer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.composer.setFixedHeight(42)
        self.send_btn = QPushButton("Generate"); self.send_btn.setObjectName("primary"); self.send_btn.setFixedSize(104, 42)
        self.send_btn.clicked.connect(self._chat_send)
        row.addWidget(self.composer, 1); row.addWidget(self.send_btn)
        v.addLayout(row)

        bottom = QHBoxLayout(); bottom.setSpacing(7)
        for label, full in SUGGESTIONS:
            b = QPushButton(label); b.setObjectName("chip"); b.setToolTip(full)
            b.clicked.connect(lambda _=False, x=full: (self.composer.setPlainText(x), self._chat_send()))
            bottom.addWidget(b)
        bottom.addSpacing(10)
        self.quick_chips = {}
        for label, fn in [("Run task", self._start_task), ("Build + train", self._full_build),
                          ("Evaluate", lambda: self._dispatch("evaluate it")),
                          ("Improve", lambda: self._dispatch("make it better"))]:
            b = QPushButton(label); b.setObjectName("ghost"); b.setEnabled(False); b.setFixedHeight(28)
            b.clicked.connect(lambda _=False, f=fn: f()); self.quick_chips[label] = b; bottom.addWidget(b)
        bottom.addStretch(1); v.addLayout(bottom)
        return deck

    def _toggle_dock(self, side: str):
        """Collapse/expand a side dock so the viewport can go full-bleed (Blender/Isaac-style)."""
        w = self._dock_left if side == "left" else self._dock_right
        w.setVisible(not w.isVisible())

    def _left_pane(self) -> QWidget:
        # The pane scrolls: outliner + 6 build-gate cards overflow at short window heights, and without a
        # scroll area Qt OVERLAPS the fixed-size widgets (the Build Gates cards painted over the outliner).
        w = QWidget(); w.setObjectName("lanePane"); w.setMinimumWidth(270); w.setMaximumWidth(350)
        outer = QVBoxLayout(w); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        inner = QWidget(); inner.setObjectName("lanePaneInner")
        v = QVBoxLayout(inner); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(7)
        v.addWidget(self._pane_title("PROJECT OUTLINER"))
        v.addWidget(self._faint("The selected robot, scene set, controller, evaluation and export artifacts."))
        self.outliner = QTreeWidget()
        self.outliner.setObjectName("outlinerTree")
        self.outliner.setHeaderLabels(["Artifact", "State"])
        self.outliner.setColumnWidth(0, 182)
        self.outliner.header().setStretchLastSection(True)
        self.outliner.setRootIsDecorated(True)
        self.outliner.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outliner.setFixedHeight(188)
        v.addWidget(self.outliner)
        self._render_outliner_empty()

        v.addWidget(self._pane_title("BUILD GATES"))
        self.lane_cards = {}
        self.lane_status_labels = {}
        self.gate_cards = {}
        lanes = [
            ("01", "Requirements", "task, constraints, success target", "Design"),
            ("02", "Parts / BOM", "actuators, sensors, limits", "Design"),
            ("03", "CAD / MJCF", "geometry, assembly, structure", "Simulate"),
            ("04", "Scene Set", "task environments and variants", "Simulate"),
            ("05", "Policy", "scripted or learned controller", "Train"),
            ("06", "Evidence", "score, failures, export ledger", "Evaluate"),
        ]
        for num, title, meta, stage in lanes:
            card = self._lane_card(num, title, meta)
            self.lane_cards.setdefault(stage, []).append(card)
            self.gate_cards[title] = card
            v.addWidget(card)
        v.addStretch(1)
        scroll.setWidget(inner)
        return w

    def _render_outliner_empty(self):
        if not hasattr(self, "outliner"):
            return
        self.outliner.clear()
        root = QTreeWidgetItem(["Virturoid Project", "open"])
        self.outliner.addTopLevelItem(root)
        for artifact, state in [
            ("Robot Genome", "none"),
            ("Scene Set", "pending"),
            ("Controller", "pending"),
            ("Evaluation", "pending"),
            ("Export Bundle", "pending"),
        ]:
            QTreeWidgetItem(root, [artifact, state])
        root.setExpanded(True)

    def _render_outliner(self, s: dict):
        if not hasattr(self, "outliner"):
            return
        self.outliner.clear()
        name = s.get("species") or s.get("name") or s.get("robot_class") or "robot"
        links = [str(x) for x in (s.get("links") or [])]
        end_effectors = [str(x) for x in (s.get("end_effectors") or [])]
        root = QTreeWidgetItem(["Virturoid Project", "active"])
        self.outliner.addTopLevelItem(root)

        robot = QTreeWidgetItem(root, [self._short(name, 28), "RobotGene"])
        QTreeWidgetItem(robot, ["Class", str(s.get("robot_class", "robot"))])
        QTreeWidgetItem(robot, ["Actuated DOF", str(s.get("dof", " - "))])
        body = QTreeWidgetItem(robot, ["Body chain", f"{len(links) or 'viewport'} links"])
        if links:
            for link in links[:10]:
                QTreeWidgetItem(body, [self._short(link, 28), "link"])
            if len(links) > 10:
                QTreeWidgetItem(body, ["more links", str(len(links) - 10)])
        for effector in end_effectors or ["tool frame"]:
            QTreeWidgetItem(robot, [self._short(effector, 28), "end effector"])

        scene = QTreeWidgetItem(root, ["Scene Set", "preview"])
        QTreeWidgetItem(scene, ["Primary scene", "ground plane"])
        QTreeWidgetItem(scene, ["Variants", "planned"])
        QTreeWidgetItem(root, ["Controller", "not trained"])
        QTreeWidgetItem(root, ["Evaluation", "not run"])
        QTreeWidgetItem(root, ["Export Bundle", "not packaged"])
        for node in (root, robot, body, scene):
            node.setExpanded(True)

    def _lane_card(self, num: str, title: str, meta: str) -> QFrame:
        card = QFrame(); card.setObjectName("laneCard")
        card.setFixedHeight(38); card.setToolTip(meta)
        lay = QHBoxLayout(card); lay.setContentsMargins(9, 6, 9, 6); lay.setSpacing(6)
        n = QLabel(num); n.setObjectName("laneNum")
        t = QLabel(title); t.setObjectName("laneTitle")
        m = QLabel("waiting"); m.setObjectName("laneMeta")
        self.lane_status_labels[card] = m
        lay.addWidget(n); lay.addWidget(t); lay.addStretch(1); lay.addWidget(m)
        return card

    def _apply_gate_states(self, states: dict[str, tuple[str, str]]):
        style_by_state = {"done": "laneDone", "active": "laneActive", "waiting": "laneCard"}
        for title, (label, state) in states.items():
            card = getattr(self, "gate_cards", {}).get(title)
            if card is None:
                continue
            status = self.lane_status_labels.get(card)
            if status is not None:
                status.setText(label)
            card.setObjectName(style_by_state.get(state, "laneCard"))
            card.style().unpolish(card); card.style().polish(card)

    def _set_preview_gate_states(self, imported: bool = False):
        self._artifact_gate_states = {
            "Requirements": ("captured", "done"),
            "Parts / BOM": ("imported" if imported else "inferred", "done"),
            "CAD / MJCF": ("loaded" if imported else "preview", "active"),
            "Scene Set": ("queued", "waiting"),
            "Policy": ("waiting", "waiting"),
            "Evidence": ("waiting", "waiting"),
        }
        self._apply_gate_states(self._artifact_gate_states)

    def _set_evaluated_gate_states(self):
        self._artifact_gate_states = {
            "Requirements": ("captured", "done"),
            "Parts / BOM": ("selected", "done"),
            "CAD / MJCF": ("built", "done"),
            "Scene Set": ("run", "done"),
            "Policy": ("tested", "done"),
            "Evidence": ("scored", "active"),
        }
        self._apply_gate_states(self._artifact_gate_states)

    def _timeline_pane(self) -> QWidget:
        # Tall enough that a typical assistant message AND its action chips are visible together (the chat is
        # the primary interaction surface) — a 106px strip clipped the chips below the fold, forcing a scroll.
        w = QFrame(); w.setObjectName("timeline"); w.setFixedHeight(176)
        v = QVBoxLayout(w); v.setContentsMargins(14, 7, 14, 8); v.setSpacing(5)
        h = QHBoxLayout(); h.setSpacing(8)
        h.addWidget(self._pane_title("RUN TIMELINE")); h.addStretch(1)
        clear = QPushButton("Clear Log"); clear.setObjectName("ghost"); clear.setFixedHeight(26)
        clear.clicked.connect(self._chat_welcome); h.addWidget(clear)
        v.addLayout(h)
        self.chat_scroll = QScrollArea(); self.chat_scroll.setObjectName("chatScroll"); self.chat_scroll.setWidgetResizable(True)
        cbody = QWidget(); cbody.setObjectName("pane")
        self.chat_layout = QVBoxLayout(cbody); self.chat_layout.setContentsMargins(6, 0, 6, 3)
        self.chat_layout.setSpacing(5); self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_scroll.setWidget(cbody); v.addWidget(self.chat_scroll, 1)
        self._chat_welcome()
        return w

    def _structure_tab(self) -> QWidget:
        w = QWidget(); w.setObjectName("pane"); v = QVBoxLayout(w); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(8)
        v.addWidget(self._pane_title("KINEMATIC STRUCTURE"))
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Body / link", "Role"])
        self.tree.setColumnWidth(0, 190); self.tree.setRootIsDecorated(True)
        v.addWidget(self.tree, 1)
        v.addWidget(self._faint("The composed body's base, links, joints and end-effector. "
                                "Compose or build a robot to populate it."))
        return w

    def _center_pane(self) -> QWidget:
        w = QWidget(); w.setObjectName("pane"); v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(5)
        hb = QHBoxLayout()
        tl = QToolButton(); tl.setObjectName("dockToggle"); tl.setText("<"); tl.setCursor(Qt.PointingHandCursor)
        tl.setToolTip("Collapse / show the design brief panel"); tl.clicked.connect(lambda: self._toggle_dock("left"))
        hb.addWidget(tl); hb.addWidget(self._pane_title("SIMULATION VIEWPORT")); hb.addStretch(1)
        hint = QLabel("left-drag orbit / right-drag pan / scroll zoom"); hint.setObjectName("faint")
        hb.addWidget(hint)
        tr = QToolButton(); tr.setObjectName("dockToggle"); tr.setText(">"); tr.setCursor(Qt.PointingHandCursor)
        tr.setToolTip("Collapse / show the inspector panel"); tr.clicked.connect(lambda: self._toggle_dock("right"))
        hb.addSpacing(8); hb.addWidget(tr); v.addLayout(hb)
        frame = QFrame(); frame.setObjectName("card"); fl = QVBoxLayout(frame); fl.setContentsMargins(1, 1, 1, 1)
        fl.addWidget(self.viewport); v.addWidget(frame, 1)
        return w

    def _right_pane(self) -> QWidget:
        self.inspector = QTabWidget(); self.inspector.setMinimumWidth(420); self.inspector.setDocumentMode(True)
        self.spec_scroll, self.spec_layout = self._scroll_panel()
        self.results_scroll, self.results_layout = self._scroll_panel()
        self.inspector.addTab(self.spec_scroll, "PROPERTIES")
        self.inspector.addTab(self._structure_tab(), "STRUCTURE")
        self.inspector.addTab(self.results_scroll, "EVIDENCE")
        self._render_spec_empty(); self._render_results_empty()
        return self.inspector

    # ============================================================ library + settings pages
    def _library_page(self) -> QWidget:
        scroll, v = self._scroll_panel(margin=28)
        v.setSpacing(12)
        v.addWidget(self._pane_title("MEMORY  /  FLYWHEEL"))
        v.addWidget(self._faint("Every robot body, learned skill, successful design and lesson is banked here and "
                                "reused across builds  -  the cross-robot species tree that compounds over time. "
                                "This is the moat: each robot a customer trains makes the next one faster to build."))
        n_designs = n_species = n_skills = 0
        species_list, designs_list, skills_list = [], [], []
        try:
            from virturoid.services.memory_db import MemoryDB
            with MemoryDB(self.workspace / "memory" / "virturoid_memory.db") as db:
                c = db.conn
                n_designs = c.execute("SELECT COUNT(*) FROM designs").fetchone()[0]
                n_species = c.execute("SELECT COUNT(*) FROM species").fetchone()[0]
                n_skills = c.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
                species_list = c.execute("SELECT name, robot_class, best_success_rate FROM species "
                                         "ORDER BY best_success_rate DESC LIMIT 6").fetchall()
                designs_list = c.execute("SELECT robot_class, prompt, success_rate FROM designs "
                                         "ORDER BY created_at DESC LIMIT 6").fetchall()
                try:
                    skills_list = c.execute("SELECT kind, task_type, success_rate FROM skills "
                                            "ORDER BY success_rate DESC LIMIT 6").fetchall()
                except Exception:  # noqa: BLE001 - skills schema is optional
                    skills_list = []
        except Exception:  # noqa: BLE001
            pass
        # headline KPI tiles (reuse the EVIDENCE KPI style so the counts read with weight, not as tiny rows)
        row = QHBoxLayout()
        row.addWidget(self._kpi("Banked designs", str(n_designs), ACCENT))
        row.addWidget(self._kpi("Species in tree", str(n_species), SIGNAL))
        row.addWidget(self._kpi("Reusable skills", str(n_skills), OK))
        rw = QWidget(); rw.setLayout(row); v.addWidget(rw)
        # always-present explainer so the flywheel reads even before anything is banked
        v.addWidget(self._flywheel_diagram())
        if species_list:
            self._section_into(v, "Top species (best task success banked)",
                               [(f"{r[0]}  /  {r[1]}", self._pct(r[2])) for r in species_list])
        if skills_list:
            self._section_into(v, "Reusable skills (transfer across bodies)",
                               [(f"{r[0]}  -  {r[1]}", self._pct(r[2])) for r in skills_list])
        if designs_list:
            self._section_into(v, "Recent designs",
                               [(self._design_label(r[1]) if r[1] else r[0], self._pct(r[2]))
                                for r in designs_list])
        if not (species_list or designs_list or skills_list):
            v.addWidget(self._faint("Empty for now  -  build a robot and train a skill, and this library fills in. "
                                    "The NEXT build of a similar robot warm-starts from the closest banked design "
                                    "and reuses banked skills/policies instead of training from scratch."))
        v.addStretch(1)
        return scroll

    def _flywheel_diagram(self) -> QFrame:
        """A compact, always-visible explainer of the compounding loop — the demo's USP, legible even on an
        empty memory DB."""
        f = QFrame(); f.setObjectName("card"); lay = QVBoxLayout(f)
        lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(8)
        h = QLabel("HOW THE FLYWHEEL COMPOUNDS"); h.setObjectName("h2"); lay.addWidget(h)
        steps = [("01", "Build", "an AI-designed body + bill of materials from the prompt"),
                 ("02", "Learn", "train a control policy in real MuJoCo physics"),
                 ("03", "Bank", "store the body, policy, skill and lessons in the species tree"),
                 ("04", "Reuse", "the next similar robot warm-starts from it  -  faster and better")]
        for num, title, desc in steps:
            r = QHBoxLayout(); r.setSpacing(10)
            b = QLabel(num); b.setStyleSheet(f"color:{ACCENT}; font-family:Consolas; font-weight:700;"); b.setFixedWidth(22)
            t = QLabel(title); t.setStyleSheet(f"color:{TEXT}; font-weight:700;"); t.setFixedWidth(58)
            d = QLabel(desc); d.setStyleSheet(f"color:{MUTED};"); d.setWordWrap(True)
            r.addWidget(b); r.addWidget(t); r.addWidget(d, 1)
            rw = QWidget(); rw.setLayout(r); lay.addWidget(rw)
        cap = QLabel("Result: every robot a customer trains makes the next one cheaper to build  -  the compounding moat.")
        cap.setStyleSheet(f"color:{SIGNAL}; font-size:11px;"); cap.setWordWrap(True); lay.addWidget(cap)
        return f

    def _settings_page(self) -> QWidget:
        scroll, v = self._scroll_panel(margin=24)
        v.addWidget(self._pane_title("STUDIO SETTINGS"))
        self._section_into(v, "Workspace", [("Project folder", str(self.project_dir)),
                                            ("Memory / flywheel", str(self.workspace / "memory"))])
        try:
            from virturoid.services import gpu_trainer as _gt
            host, py = _gt._HOST, _gt._PY
        except Exception:  # noqa: BLE001
            host, py = " - ", " - "
        self._section_into(v, "GPU training  /  default when reachable", [
            ("Status (at launch)", "ready" if self._gpu_ok else "offline -> on-device fallback"),
            ("Host", host), ("Python", py), ("Default iterations", "80"), ("Engine", "MJX PPO (CUDA)")])
        self._section_into(v, "Compute", [("Simulation + design", "Local CPU (MuJoCo)"),
                                          ("Policy training", "GPU (MJX PPO) by default; on-device CPU fallback"),
                                          ("Physics engine", "MuJoCo")])
        self._section_into(v, "Appearance", [("Theme", "Native robotics studio"), ("Accent", "Signal orange + telemetry teal")])
        v.addWidget(self._faint("Virturoid Studio is the primary app. It calls the build/evaluate engine directly "
                                "from this native desktop process; the browser server is only a debugging surface."))
        v.addStretch(1)
        return scroll

    def _build_statusbar(self):
        self.backend_chip = _chip("MuJoCo physics", OK)
        self.gpu_chip = _chip("GPU: checking...", MUTED)
        path = QLabel("  " + str(self.project_dir)); path.setObjectName("faint")
        self.statusBar().addWidget(path)
        self.statusBar().addPermanentWidget(self.gpu_chip)
        self.statusBar().addPermanentWidget(self.backend_chip)

    def _on_gpu_ping(self, ok: bool):
        self._gpu_ok = bool(ok)
        self.gpu_chip.setText("GPU ready" if ok else "GPU offline")
        self.gpu_chip.setStyleSheet(_chip_qss(OK if ok else FAINT))

    # ============================================================ small helpers
    def _scroll_panel(self, margin: int = 14):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        body = QWidget(); body.setObjectName("pane")
        lay = QVBoxLayout(body); lay.setContentsMargins(margin, margin, margin, margin); lay.setSpacing(11)
        lay.setAlignment(Qt.AlignTop); scroll.setWidget(body)
        return scroll, lay

    def _pane_title(self, text: str) -> QLabel:
        lab = QLabel(text); lab.setObjectName("paneTitle"); return lab

    def _hr(self) -> QFrame:
        f = QFrame(); f.setObjectName("hr"); return f

    def _faint(self, text: str) -> QLabel:
        lab = QLabel(text); lab.setObjectName("faint"); lab.setWordWrap(True); return self._hfw(lab)

    def _log(self, text: str, color: str = MUTED):
        lab = QLabel(text); lab.setObjectName("sysLine"); lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
        lab.setStyleSheet(f"color: {color}; font-size: 12px; background: transparent;")
        self.chat_layout.addWidget(lab); self._chat_scroll_bottom()

    @staticmethod
    def _clear(layout):
        while layout.count():
            it = layout.takeAt(0)
            widget = it.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    @staticmethod
    def _hfw(w):
        """Make a widget report height-for-width to its parent layout so wrapped text isn't clipped."""
        sp = w.sizePolicy(); sp.setHeightForWidth(True); w.setSizePolicy(sp); return w

    def _set_pill(self, color: str):
        self.proj_pill.setStyleSheet(_chip_qss(color))

    def _pill(self, text: str):
        """Set the project pill, elided to its max width with the full name in the tooltip  -  so a long
        species name truncates with '...' instead of pushing the top-bar buttons off-screen."""
        from PySide6.QtGui import QFontMetrics
        self.proj_pill.setToolTip(text)
        self.proj_pill.setText(QFontMetrics(self.proj_pill.font()).elidedText(text, Qt.ElideRight, 214))

    def _set_preview_pill(self, summary: dict):
        name = summary.get("species") or summary.get("name") or summary.get("robot_class") or "robot"
        robot_class = str(summary.get("robot_class") or name)
        dof = summary.get("dof")
        display_name = robot_class if len(str(name)) > 18 else str(name)
        label = f"Preview: {display_name}"
        if dof not in (None, " - "):
            label += f" / {dof} DOF"
        self._pill(label)
        self.proj_pill.setToolTip(f"{name} / preview" + (f" / {dof} DOF" if dof not in (None, " - ") else ""))
        self._set_pill(SIGNAL)

    def _on_new(self):
        self.composed_gene = None; self.composed_meta = None; self.composed_prompt = ""
        self._last_policy = None
        self._is_busy = False; self._auto = False; self._artifact_gate_states = None
        self._pill("No robot yet"); self._set_pill(MUTED)
        self.composer.clear(); self.send_btn.setEnabled(True); self.send_btn.setText("Generate")
        self.viewport.show_message("Describe a robot in the design brief - it renders here in real physics.")
        self._render_spec_empty(); self._render_results_empty(); self.tree.clear(); self._render_outliner_empty()
        self._go_section(0); self.inspector.setCurrentIndex(0); self._set_stage("Design")
        self._chat_welcome(); self._update_quick()

    # ---- import an existing CAD/robot model (MJCF / URDF) and iterate on it ----
    def _import_model(self):
        if self._busy():
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Import a robot model", "",
                                              "Robot models (*.xml *.mjcf *.urdf);;All files (*)")
        if not path:
            return
        from virturoid.services.model_import import import_model
        self._chat_user(f"Import {Path(path).name}")
        imp = import_model(path)
        if not imp.get("ok"):
            self._chat_ai(f"<span style='color:{BAD}'>Couldn't import that model.</span> {imp.get('note', '')}")
            return
        self.composed_gene = imp["mjcf"]; self.composed_meta = imp; self.composed_prompt = imp["name"]
        self._last_policy = None
        self.viewport.load_static(imp["mjcf"], {"robot_class": "imported", "dof": imp["actuated"]})
        self._populate_tree_from_model(imp["name"])
        self._render_spec({"species": imp["name"], "name": imp["name"], "robot_class": "imported model",
                           "dof": imp["actuated"], "links": [], "end_effectors": [], "valid": True,
                           "design_source": "imported"}, imp["name"])
        self._render_outliner({"species": imp["name"], "name": imp["name"], "robot_class": "imported model",
                               "dof": imp["actuated"], "links": [], "end_effectors": [], "valid": True,
                               "design_source": "imported"})
        self._set_preview_pill({"name": imp["name"], "robot_class": "imported", "dof": imp["actuated"]})
        self._set_preview_gate_states(imported=True)
        self.inspector.setCurrentIndex(0); self._set_stage("Simulate")
        learnable = imp.get("free_base") and imp.get("actuated", 0) > 0
        if learnable:
            from PySide6.QtCore import QTimer
            self._chat_ai(f"Imported <b>{imp['name']}</b>  -  {imp['parts']} parts, {imp['actuated']} actuated DOF. "
                          f"It's in the viewport. I'll learn control for it now, then you can keep iterating.",
                          chips=[("Learn to move", self._learn)])
            QTimer.singleShot(700, self._learn)
        else:
            if imp.get("actuated", 0) > 0:                 # has motors but a fixed base -> simulate, no gait
                extra = ("It has motors but a fixed base, so it's loaded here for simulation  -  learning a "
                         "locomotion gait needs a free-floating base.")
            else:
                extra = imp["note"] + "."
            self._chat_ai(f"Imported <b>{imp['name']}</b>  -  {imp['parts']} parts, {imp['actuated']} actuated DOF. "
                          f"{extra} It's in the viewport.")
        self._update_quick()

    # ============================================================ chat surface
    def _chat_welcome(self):
        self._clear(self.chat_layout); self._status_lab = None
        box = QFrame(); box.setObjectName("welcome"); bl = QVBoxLayout(box)
        bl.setContentsMargins(11, 6, 11, 6); bl.setSpacing(2)
        t = QLabel("Ready - no run yet"); t.setObjectName("welcomeTitle"); bl.addWidget(t)
        bl.addWidget(self._faint("Events, training notes, failures and export proof appear here as the loop runs."))
        self.chat_layout.addWidget(box)
        self._welcome_shown = True

    def _chat_user(self, text: str):
        if getattr(self, "_welcome_shown", False):       # first message replaces the welcome (Claude-style)
            self._clear(self.chat_layout); self._welcome_shown = False
        row = QFrame(); rl = QVBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        bub = QFrame(); bub.setObjectName("bubbleUser"); bub.setMaximumWidth(440)
        bl = QVBoxLayout(bub); bl.setContentsMargins(13, 9, 13, 9)
        lab = QLabel(text); lab.setObjectName("bubbleText"); lab.setWordWrap(True); lab.setTextFormat(Qt.PlainText)
        self._hfw(lab); bl.addWidget(lab)
        self._hfw(bub); self._hfw(row)
        rl.addWidget(bub, 0, Qt.AlignRight)
        self.chat_layout.addWidget(row); self._chat_scroll_bottom()

    def _chat_ai(self, html: str, chips=None):
        row = QFrame(); rl = QVBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(7)
        head = QHBoxLayout(); head.setSpacing(7)
        av = QLabel("V"); av.setObjectName("aiAvatar"); nm = QLabel("Virturoid"); nm.setObjectName("aiName")
        head.addWidget(av); head.addWidget(nm); head.addStretch(1); rl.addLayout(head)
        lab = QLabel(html); lab.setObjectName("aiText"); lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
        self._hfw(lab); rl.addWidget(lab)
        if chips:
            cr = QHBoxLayout(); cr.setSpacing(6); cr.setContentsMargins(0, 2, 0, 0)
            for label, fn in chips:
                b = QPushButton(label); b.setObjectName("chip")
                b.clicked.connect(lambda _=False, f=fn: f()); cr.addWidget(b)
            cr.addStretch(1); rl.addLayout(cr)
        self._hfw(row)
        self.chat_layout.addWidget(row); self._chat_scroll_bottom()
        return lab

    def _chat_status(self, text: str):
        self._status_lab = self._chat_ai(f"<span style='color:{MUTED}'>* {text}</span>")
        return self._status_lab

    def _chat_scroll_bottom(self):
        from PySide6.QtCore import QTimer
        bar = self.chat_scroll.verticalScrollBar()
        # Scroll twice: once on the next tick, and again after word-wrap/height-for-width has settled, so the
        # newest message AND its chips are actually pinned to the bottom (a single 0ms tick can fire too early).
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))
        QTimer.singleShot(40, lambda: bar.setValue(bar.maximum()))

    def _suggest(self):
        if self.composed_gene is not None and not self.built:
            return [("Run task", self._start_task), ("Build + train", self._full_build)]
        if self.built:
            return [("Evaluate", lambda: self._dispatch("evaluate it")),
                    ("Improve", lambda: self._dispatch("make it better")),
                    ("Export", lambda: self._dispatch("export the package"))]
        return None

    def _update_quick(self):
        b = getattr(self, "_is_busy", False)
        has_gene = self.composed_gene is not None and not isinstance(self.composed_gene, str)  # str = imported model
        self.quick_chips["Run task"].setEnabled(not b and has_gene)
        self.quick_chips["Build + train"].setEnabled(not b and (has_gene or self.built))
        self.quick_chips["Evaluate"].setEnabled(not b and self.built)
        self.quick_chips["Improve"].setEnabled(not b and self.built)
        if hasattr(self, "open_pkg_btn"):
            self.open_pkg_btn.setEnabled(self.built and Path(self.project_dir).exists())

    def _open_package(self):
        """Reveal the built package folder on disk so a reviewer can open the real artifacts (the EXPORT stage's
        deliverable). No-op with an honest note until a build has actually written the package."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        pkg = Path(self.project_dir)
        if not self.built or not pkg.exists():
            self._chat_ai("There's no built package yet — describe a robot and use <b>Build + Train</b> first, "
                          "then I can open its package folder.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(pkg.resolve())))
        self._chat_ai(f"Opened the package folder: <code>{pkg}</code> — Robot Genome, CAD (STEP/STL), generated "
                      f"scenes, BOM, reports (incl. the honest readiness ledger), and the ROS2 bundle are there.")

    def _busy(self) -> bool:
        return getattr(self, "_is_busy", False)

    def _set_busy(self, b: bool):
        self._is_busy = b
        self.composer.setReadOnly(b)
        self.send_btn.setEnabled(not b); self.send_btn.setText("..." if b else "Generate")
        if hasattr(self, "build_pkg_btn"):
            self.build_pkg_btn.setEnabled(not b)
        self._update_quick()

    # ============================================================ chat routing -> backend
    def _chat_send(self):
        self._dispatch(self.composer.toPlainText())

    def _dispatch(self, text: str):
        text = (text or "").strip()
        if not text or self._busy():
            return
        from virturoid.services.agent import parse_intent
        self.composer.clear(); self._chat_user(text)
        low = text.lower(); intent, _ = parse_intent(text)
        if any(k in low for k in ("run task", "run the task", "plan the task", "do the task")):
            if self.composed_gene is None:
                self._chat_ai("Describe a robot first, then I'll plan and run its task."); return
            self._start_task(); return
        # learn-on-request: a legged/humanoid body needs a learned gait  -  train it now
        if (intent == "train" or any(k in low for k in ("learn to", "teach it", "make it walk", "make it move",
                "walk", "locomot", "gait", "gpu"))) and self.composed_gene is not None and self._is_legged():
            self._learn(use_gpu=("gpu" in low)); return
        explicit_build = any(k in low for k in ("build it", "for real", "package", "co-design", "co design"))
        if intent == "build" and self.composed_gene is None and not explicit_build:
            self._start_compose(text); return                      # first look: instant body
        if intent == "build" or explicit_build:
            self._start_build(self.composed_prompt or text); return
        if intent in ("evaluate", "iterate", "perceive", "train", "export", "status", "adjust_and_rebuild"):
            if not self.built and intent in ("evaluate", "iterate", "perceive", "export"):
                self._chat_ai(f"There's no built robot yet  -  describe one and I'll build it, then I can {intent} it.")
                return
            self._start_job(text); return
        self._start_compose(text)                                  # default: treat as a robot description

    # ---- compose (fast body preview) ----
    def _start_compose(self, text: str):
        if self.compose_worker and self.compose_worker.isRunning():
            return
        self.composed_prompt = text
        self._auto = True                  # a typed request runs the whole pipeline automatically
        self._set_busy(True); self._chat_status("Designing the body...")
        self.viewport.show_message("Designing the body...")
        self.compose_worker = ComposeWorker(text)
        self.compose_worker.ready.connect(self._on_composed)
        self.compose_worker.failed.connect(self._on_preview_failed)
        self.compose_worker.start()

    def _on_composed(self, summary: dict, xml: str):
        self._set_busy(False); self._status_lab = None
        self._last_policy = None
        self.composed_prompt = summary.get("_prompt", self.composed_prompt)
        if summary.get("design_source") == "real":              # (A) a REAL production model  -  carried as MJCF
            res = summary["_real"]
            self.composed_gene = res["mjcf"]                    # string body, like an imported model
            self.composed_meta = {"name": res["label"], "free_base": res["free_base"],
                                  "actuated": res["actuated"], "parts": res["bodies"]}
            self.viewport.load_static(xml, summary)
            self._populate_tree_from_model(res["label"])
            self._render_spec(summary, self.composed_prompt)
            self._render_outliner(summary)
            self._set_preview_pill(summary)
            self._set_preview_gate_states(imported=False)
            self.inspector.setCurrentIndex(0); self._set_stage("Simulate")
            auto = self._auto; self._auto = False
            legged = self._is_legged()
            train_hint = (" Press <b>Learn to move</b> to train a control policy on the GPU (a few minutes; live "
                          "progress below)." if (legged and self._gpu_ok) else
                          " Press <b>Learn to move</b> to train a control policy on this body." if legged else "")
            self._chat_ai(f"Built <b>{res['label']}</b>  -  a real production robot: {res['bodies']} parts, "
                          f"{res['actuated']} actuators, {res['meshes']} meshes (real geometry + inertias). "
                          f"It's in the viewport.{train_hint}",
                          chips=([("Learn to move", self._learn)] if legged else None))
            self._update_quick()
            return
        self.composed_gene = summary.get("_gene")
        self.composed_meta = None
        self.viewport.load_static(xml, summary)
        self._render_spec(summary, self.composed_prompt); self._populate_tree(summary); self._render_outliner(summary)
        self._set_preview_pill(summary)
        self._set_preview_gate_states(imported=False)
        self.inspector.setCurrentIndex(0); self._set_stage("Simulate")
        cls = summary.get("robot_class", "robot"); dof = summary.get("dof", 0)
        parts = len(summary.get("links", []))
        valid = "valid kinematic tree" if summary.get("valid") else "kinematic tree needs fixing"
        auto = self._auto; self._auto = False
        from PySide6.QtCore import QTimer
        if self._is_legged():
            # Training is an EXPLICIT, observed step  -  never silently commit a typed prompt to a multi-minute
            # GPU run (that leaves the studio looking frozen with no bounded progress). Show the body now; the
            # default "Learn to move" is the fast on-device recipe (~1 min, bounded, live progress), and the
            # longer GPU PPO is a clearly-labelled opt-in for a stronger gait.
            chips = [("Learn to move", lambda: self._learn(False))]
            if self._gpu_ok:
                chips.append(("Train on GPU (stronger, a few min)", lambda: self._learn(True)))
            chips.append(("Improve design", lambda: self._dispatch("make it better")))
            self._chat_ai(f"Designed a <b>{cls}</b>  -  {dof} actuated DOF, {parts} parts, {valid}. It's in the "
                          f"viewport; full spec on the right. Press <b>Learn to move</b> to train a gait on this "
                          f"body and watch the replay" + (" — or <b>Train on GPU</b> for a stronger, longer run."
                          if self._gpu_ok else "."),
                          chips=chips)
        else:
            if not self._class_was_inferred(self.composed_prompt, cls):
                self._chat_ai("I couldn't infer a clear robot type from that prompt, so I defaulted to a tabletop "
                              "<b>arm</b>. For a specific robot, name it and a task — e.g. “a quadruped that "
                              "walks”, “a rover that navigates”, or “an arm that sorts blocks”.")
            nxt = " Now planning + running its task..." if auto else " I can run its task whenever you're ready."
            self._chat_ai(f"Designed a <b>{cls}</b>  -  {dof} actuated DOF, {parts} parts, {valid}. It's in the "
                          f"viewport; full spec on the right.{nxt}",
                          chips=[("Run task", self._start_task), ("Build + train", self._full_build),
                                 ("Improve design", lambda: self._dispatch("make it better"))])
            if auto:
                QTimer.singleShot(700, self._start_task)   # the general task layer dispatches the right skill
        self._update_quick()

    def _class_was_inferred(self, prompt: str, cls: str) -> bool:
        """True if the robot_class came from a real signal — a prompt class-keyword or an LLM — rather than the
        OFFLINE composer's silent manipulator default. Used to honestly flag 'I couldn't infer a robot type'
        instead of confidently presenting an arm for an unrecognized prompt (e.g. 'a banana')."""
        import os
        try:
            from virturoid.services.morphology_composer import (
                _GRASP_WORDS, _HUMANOID_WORDS, _LEGGED_WORDS, _MOBILE_WORDS,
            )
            p = (prompt or "").lower()
            if "arm" in p or any(w in p for w in (*_HUMANOID_WORDS, *_LEGGED_WORDS, *_MOBILE_WORDS, *_GRASP_WORDS)):
                return True                                   # a class keyword matched
        except Exception:  # noqa: BLE001
            return True                                       # never block the flow on an import hiccup
        # no class keyword: only an LLM backend could have classified it; offline -> it's the manipulator default
        return os.environ.get("VIRTUROID_LLM_BACKEND", "off").lower() not in ("off", "", "mock")

    def _on_preview_failed(self, err: str):
        self._set_busy(False); self._status_lab = None
        self._chat_ai(f"<span style='color:{BAD}'>I couldn't compose that robot.</span> {err}")
        self.viewport.show_message("Could not compose this robot:\n" + err)

    # ---- full build / evaluate / improve / ... (the agent) ----
    def _full_build(self):
        if self._busy():
            return
        if not self.composed_prompt:
            self._chat_ai("Describe a robot first  -  then I'll build &amp; train it."); return
        self._chat_user("Build and train it"); self._start_build(self.composed_prompt)

    def _start_build(self, prompt: str):
        msg = prompt if prompt.lower().startswith("build ") else ("build " + prompt)
        self._run_message(msg, status="Designing, validating and co-designing the robot...")

    def _start_job(self, text: str):
        from virturoid.services.agent import parse_intent
        labels = {"evaluate": "Running the task in real physics...", "iterate": "Re-running the build + co-optimizing...",
                  "perceive": "Adding perception and re-running...", "train": "Setting up policy training...",
                  "export": "Packaging the robot...", "status": "Checking status...",
                  "adjust_and_rebuild": "Adjusting and rebuilding..."}
        self._run_message(text, status=labels.get(parse_intent(text)[0], "Working..."))

    def _run_message(self, message: str, status: str = "Working..."):
        if self.worker and self.worker.isRunning():
            return
        self._set_busy(True); self._chat_status(status)
        self.worker = JobWorker(self.workspace, self.project_dir, self.agent, message)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, ev: dict):
        msg = ev.get("message") or ev.get("stage") or ""
        if self._status_lab is not None and msg:
            self._status_lab.setText(f"<span style='color:{MUTED}'>* {msg}</span>"); self._chat_scroll_bottom()

    def _on_done(self, result: dict):
        self._set_busy(False); self._status_lab = None
        self.refresh_project()
        self._chat_ai(result.get("message", "Done."), chips=self._suggest())
        if self.built and result.get("intent") in {"build", "iterate", "adjust_and_rebuild", "evaluate"}:
            self._load_episode(0)

    def _on_failed(self, err: str):
        self._set_busy(False); self._status_lab = None
        self._chat_ai(f"<span style='color:{BAD}'>Something went wrong.</span> {err}")

    # ---- learn-on-request: train control for THIS body, replay, bank + commit species tips ----
    def _is_legged(self) -> bool:
        if self.composed_gene is None:
            return False
        if isinstance(self.composed_gene, str):           # imported model: learnable if it can move + has motors
            m = self.composed_meta or {}
            return bool(m.get("free_base") and m.get("actuated", 0) > 0)
        cls = (self.composed_gene.robot_class or "").lower()
        if cls in {"humanoid", "biped", "quadruped", "legged"}:
            return True
        try:
            from virturoid.services.task_matched_eval import robot_kind
            return robot_kind(self.composed_gene) == "legged"
        except Exception:  # noqa: BLE001
            return False

    def _recall_policy(self):
        if self.composed_gene is None or isinstance(self.composed_gene, str):   # imported models: no gene to key on
            return None
        try:
            from virturoid.services.memory_db import MemoryDB
            from virturoid.services.policy_flywheel import recall_morph_policy
            with MemoryDB(self.workspace / "memory" / "virturoid_memory.db") as db:
                return recall_morph_policy(self.composed_gene, db, task_type="locomotion")
        except Exception:  # noqa: BLE001
            return None

    def _learn(self, use_gpu=None):
        if self.composed_gene is None:
            self._chat_ai("Describe a robot first, then I'll learn to control it."); return
        if self._busy():
            return
        imported = isinstance(self.composed_gene, str)    # imported MJCF  -  trains on GPU via --mjcf-file, or CPU
        if use_gpu is None:
            use_gpu = self._gpu_ok                          # GPU is the default backend whenever the box is reachable
        # iterate: warm-start from THIS body's last learned policy (compounds), else banked tips for the species
        if use_gpu:
            warm = None
        elif self._last_policy is not None:
            warm = self._last_policy
        else:
            warm = None if imported else self._recall_policy()
        species = (self.composed_meta or {}).get("name") if imported else None
        self._chat_user("Train on the GPU" if use_gpu else "Learn to move")
        self._set_busy(True)
        if use_gpu:
            self._chat_status("Shipping this body to the GPU and training a control policy (a few minutes)...")
        else:
            self._chat_status("Warm-starting from tips banked for similar robots, then refining on this body..."
                              if warm is not None else "Learning to move  -  training a control policy on this body...")
        self.learn_worker = LearnWorker(self.composed_gene, self.workspace, warm_start=warm, use_gpu=use_gpu,
                                        species=species)
        self.learn_worker.note.connect(self._on_learn_note)
        self.learn_worker.done.connect(self._on_learned)
        self.learn_worker.failed.connect(lambda e: (self._set_busy(False),
                                                    self._chat_ai(f"<span style='color:{BAD}'>Training failed.</span> {e}")))
        self.learn_worker.start()

    def _on_learn_note(self, m: str):
        if self._status_lab is not None:
            self._status_lab.setText(f"<span style='color:{MUTED}'>* {m}</span>"); self._chat_scroll_bottom()

    def _on_learned(self, res: dict, view: dict, xml: str):
        self._set_busy(False); self._status_lab = None
        self._last_policy = res.get("policy")              # iterate: next "Train more" compounds on this
        self.viewport.load_episode(view, xml)              # replay the learned motion
        self._set_stage("Train")
        sp = res.get("species", "this robot")
        sc, base = res.get("score", 0.0), res.get("baseline", 0.0)
        warm = " (warm-started from tips banked for similar robots)" if res.get("warm_started") else ""
        gpu = " on the GPU" if res.get("backend") == "gpu" else ""
        banked = (f"banked the policy + saved tips to memory, so the next {sp} warm-starts from it"
                  if res.get("skill_id") else f"(couldn't bank it: {res.get('bank_error', 'memory unavailable')})")
        if isinstance(self.composed_gene, str):            # imported model  -  iterate by training further
            chips = [("Train more", lambda: self._learn())]
        else:
            chips = [("Run task", self._start_task), ("Build + train", self._full_build),
                     ("Train more", lambda: self._learn())]
        _fm = res.get("forward_m")
        fwd = f"; it travelled {_fm:.2f} m forward over the deploy horizon" if isinstance(_fm, (int, float)) else ""
        # When a body warm-starts from its banked gait and is ALREADY near-converged, a flat reward is the
        # flywheel paying off (reuse), not a failed training run — frame it that way instead of "improved little".
        if res.get("warm_started") and (sc - base) <= 0.03 and res.get("backend") != "gpu":
            self._chat_ai(f"Warm-started <b>{sp}</b> from the gait banked for this species — it was already "
                          f"near its converged quality (gait reward {sc:.2f}), so this pass reused that policy "
                          f"instead of re-learning from scratch{fwd}. <b>That reuse is the flywheel</b>: the next "
                          f"{sp} starts here, not at zero. The replay is in the viewport. Use <b>Train more</b> "
                          f"to push it further on the GPU.", chips=chips)
        else:
            note = ""
            if sc - base <= 0.03 and res.get("backend") != "gpu":
                note = (" It improved only a little in this quick on-device pass  -  balance-heavy bodies "
                        "(humanoids) need the GPU trainer, which banks into the same flywheel.")
            self._chat_ai(f"Trained a locomotion policy{gpu} for <b>{sp}</b>{warm}  -  gait reward {sc:.2f} (was "
                          f"{base:.2f} at the start){fwd}. I {banked}. The replay is in the viewport.{note}",
                          chips=chips)

    # ---- general task layer ----
    def _has_banked_locomotion(self) -> bool:
        try:
            from virturoid.services.learn_locomotion import banked_policy_for
            return banked_policy_for(self.composed_gene, models_dir="models") is not None
        except Exception:  # noqa: BLE001
            return False

    def _start_task(self):
        if self.composed_gene is None or self._busy():
            return
        # A walker's "task" IS its gait. Route a legged body to the learn-on-demand recipe (which trains a REAL
        # walking policy and replays it in the viewport) instead of the scripted task eval — which, with no banked
        # policy, drives the quad backward and scores ~0. Once a policy is banked, the task eval reuses it (recipe).
        if self._is_legged() and not self._has_banked_locomotion():
            self._learn()
            return
        if self.task_worker and self.task_worker.isRunning():
            return
        self._set_busy(True); self._chat_status("Planning the task, verifying it's feasible, and running it...")
        self.task_worker = TaskWorker(self.composed_prompt, self.composed_gene)
        self.task_worker.done.connect(self._on_task_done)
        self.task_worker.failed.connect(lambda e: (self._set_busy(False),
                                                   self._chat_ai(f"<span style='color:{BAD}'>Task error.</span> {e}")))
        self.task_worker.start()

    def _on_task_done(self, r: dict):
        self._set_busy(False); self._status_lab = None
        # The general task layer recorded a replay of whatever skill it dispatched (maze drive / walk / nav).
        # Load it FIRST so the score is shown WITH the visible run (the truth gate), never over a frozen viewport.
        view, xml = r.get("view"), r.get("model_xml")
        if view and view.get("frames") and xml:
            try:
                self.viewport.load_episode(view, xml)
            except Exception as exc:  # noqa: BLE001
                self.viewport.show_message("Could not display this run:\n" + str(exc))
        self._render_task_results(r)
        if not r.get("feasible", False):
            gaps = "; ".join(r.get("issues", [])[:3]) or "the morphology doesn't support the required skills"
            self._chat_ai(f"This robot <b>can't do that task</b> as built  -  {gaps}. I can improve the design "
                          f"to close the gaps.", chips=[("Improve design", lambda: self._dispatch("make it better"))])
            return
        if not self.viewport.has_live_episode():
            self._chat_ai("I checked feasibility and planned the skills, but I did <b>not</b> run a visible "
                          "simulation for this task, so I'm not claiming a success score. Use <b>Build + train</b> "
                          "(or 'Run task' on a legged robot) to actually simulate it.", chips=self._suggest())
            return
        # Be honest about the outcome: a 0.0 score is NOT a "partial success" — say it ran but didn't meet the
        # goal, so the chat never contradicts the Evidence panel (which shows score / goal-predicates / failure).
        score = float(r.get("score") or 0.0)
        if r.get("success"):
            verdict = "succeeded"
        elif score > 0.0:
            verdict = "partially succeeded"
        else:
            verdict = "ran, but didn't complete the goal"
        self._chat_ai(f"Task <b>{verdict}</b>  -  score {score:.2f}. Skills: "
                      f"{', '.join(r.get('steps_planned', [])) or ' - '}. Full breakdown is in the Evidence tab.",
                      chips=([("Build + train", self._full_build)] if not self.built else self._suggest()))

    def _render_task_results(self, r: dict):
        self._clear_results()
        self.inspector.setCurrentIndex(2); self._set_stage("Evaluate")
        self._set_evaluated_gate_states()
        live = self.viewport.has_live_episode()
        head = QLabel("TASK RESULT" if live else "FEASIBILITY ANALYSIS  (no simulation shown)")
        head.setObjectName("h2"); self.results_layout.addWidget(head)
        rows = [("Feasible", "yes" if r.get("feasible") else "no  -  see gaps"),
                ("Planned skills", ", ".join(r.get("steps_planned", [])) or " - "),
                ("Task source", r.get("task_source", " - "))]
        gm = r.get("grasp_model")
        if live:                                            # a score is only shown when a real episode played
            rows += [("Goal predicates met", f"{r.get('goal_met', 0)}/{r.get('goal_total', 0)}"),
                     ("Score", str(r.get("score", 0.0)))]
            # HONEST grasp disclosure right next to the score (never let the KPI read as a real grasp when it's a pin)
            if gm == "contact":
                rows.append(("Grasp", "real contact  -  fingers close + hold by friction (no pin)"))
            elif gm == "idealized_pin":
                rows.append(("Grasp", "idealized  -  block pinned to the gripper on contact"))
            # Show the ACTUAL placement rate (not just the pass/fail goal score) so a lenient goal predicate
            # can never let "Score 1.0" read as a perfect run when the real rate is partial.
            gsr = r.get("grasp_success_rate")
            if gm in ("contact", "idealized_pin") and gsr is not None:
                rows.append(("Objects placed", f"{gsr:.0%} of blocks"
                             + ("  (real friction grasp)" if gm == "contact" else "  (idealized pin)")))
        self._section("Outcome", rows)
        if live and gm == "idealized_pin":
            self.results_layout.addWidget(self._faint("This run uses an IDEALIZED engagement (the block is pinned "
                "to the gripper on contact). A real friction grasp is certified separately  -  see the Evidence "
                "ledger for its measured rate; the default arm tasks run the real contact grasp, not the pin."))
        if not live:
            self.results_layout.addWidget(self._faint("This is a feasibility / plan check, NOT a simulated run  -  "
                "no episode played in the viewport, so no task score is claimed. Use 'Build + train' (or 'Run task' "
                "on a legged robot) to actually simulate it."))
        if r.get("issues"):
            self.results_layout.addWidget(self._faint("Why it can't (gaps): " + "; ".join(r["issues"][:4])))

    # ---- project / results ----
    def refresh_project(self):
        p = self._project_summary()
        self.built = bool(p.get("built"))
        self._update_quick()
        if not self.built:
            self._pill("No robot yet"); self._set_pill(MUTED); return
        auto, ev = p.get("autonomy", {}), p.get("evaluation", {})
        rate = ev.get("success_rate", auto.get("final_success_rate"))
        self._pill(f"{auto.get('species', 'robot')}  /  {self._pct(rate)}")
        self._set_pill(OK if auto.get("succeeded") else WARN)
        self._render_results(p)
        self.inspector.setCurrentIndex(2); self._set_stage("Evaluate")
        self._set_evaluated_gate_states()

    def _load_episode(self, scene_index: int = 0):
        if not self.built:
            return
        if self.render_worker and self.render_worker.isRunning():
            return
        self.viewport.show_message("Running episode in MuJoCo...")
        self.render_worker = EpisodeWorker(self.project_dir, scene_index or 0)
        self.render_worker.ready.connect(self._on_episode)
        self.render_worker.failed.connect(lambda e: self.viewport.show_message(
            "3D render unavailable on this machine:\n" + e + "\n(The build, results, and metrics still work.)"))
        self.render_worker.start()

    def _on_episode(self, view: dict, xml: str):
        try:
            self.viewport.load_episode(view, xml)
            # the episode is live now -> re-render Evidence so the (honest) score appears WITH the visible replay,
            # not before it. If load produced no frames, has_live_episode() stays False and the score stays withheld.
            if self.built:
                self._render_results(self._project_summary())
        except Exception as exc:  # noqa: BLE001 - a render/model error must never CLOSE the app
            self.viewport.show_message("Could not display this episode:\n" + str(exc))

    # ---- results rendering ----
    def _clear_results(self):
        while self.results_layout.count():
            it = self.results_layout.takeAt(0)
            widget = it.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def _render_results_empty(self):
        self._clear_results()
        t = QLabel("Results appear here"); t.setObjectName("muted"); t.setStyleSheet(f"color:{MUTED}; font-weight:600;")
        s = self._faint("Build and evaluate a robot to see its task success, what the AI chose to build, "
                        "where it struggled, and proof it ran in real physics.")
        self.results_layout.addWidget(t); self.results_layout.addWidget(s)

    # ---- parsed spec (the trust bridge between language and the generated robot) ----
    def _render_spec_empty(self):
        self._clear(self.spec_layout)
        self.spec_layout.addWidget(self._inspector_banner(
            "No selection", "Project properties", [("IDLE", "warn"), ("DESIGN", "metric")]
        ))
        self.spec_layout.addWidget(self._property_group("Selection", [
            ("Active object", "none"),
            ("Next action", "Generate a robot from Mission Command"),
            ("Viewport", "waiting for a composed body"),
        ]))
        self.spec_layout.addWidget(self._property_group("Required artifacts", [
            ("Requirements", "pending"),
            ("Parts / BOM", "pending"),
            ("CAD / MJCF", "pending"),
            ("Scene set", "pending"),
            ("Evaluation", "pending"),
            ("Export bundle", "pending"),
        ]))
        self._sync_scroll_height(self.spec_layout)

    def _render_spec(self, s: dict, prompt: str):
        self._clear(self.spec_layout)
        valid = bool(s.get("valid", False))
        dof = s.get("dof", " - ")
        robot_class = str(s.get("robot_class", "robot"))
        name = str(s.get("species") or s.get("name") or robot_class)
        source = _DESIGN_SOURCE_LABEL.get(s.get("design_source"), "offline template")
        chips = [("VALID" if valid else "CHECK", "ok" if valid else "warn"), (f"{dof} DOF", "metric"), (source, "metric")]
        self.spec_layout.addWidget(self._inspector_banner(name, f"{robot_class} / Robot Genome", chips))

        ee = s.get("end_effectors", []) or []
        links = [str(x) for x in (s.get("links") or [])]
        chain = " -> ".join(links) if links else "viewport body tree"
        goal = self._short(prompt or " - ", 86)
        task_route = self._task_route(prompt)
        self.spec_layout.addWidget(self._property_group("Environment and perception", [
            ("Training scenes", "ground plane preview; randomized scene set queued"),
            ("Vision rig", "RGB, depth and LiDAR sensor slots planned"),
            ("World model", "object goals, collision geometry and task predicates"),
            ("Domain randomization", "spawn pose, lighting and texture hooks queued"),
        ]))
        self.spec_layout.addWidget(self._property_group("Robot", [
            ("Class", robot_class),
            ("Species", name),
            ("Design source", source),
            ("Validation", "kinematic tree valid" if valid else "needs kinematic review"),
        ]))
        self.spec_layout.addWidget(self._property_group("Kinematics", [
            ("Actuated DOF", str(dof)),
            ("Body chain", self._short(chain, 120)),
            ("End effector", ", ".join(str(x) for x in ee) or "tool frame"),
            ("Sensing", "joint position + velocity"),
        ]))
        self.spec_layout.addWidget(self._property_group("Task binding", [
            ("Goal", goal),
            ("Skill route", task_route),
            ("Scene policy", "primary scene plus randomized variants"),
        ]))
        self.spec_layout.addWidget(self._property_group("Simulation", [
            ("Model format", "MJCF preview"),
            ("Physics", "MuJoCo"),
            ("Scene", "ground plane preview; generator queued"),
            ("Vision inputs", "camera slots planned; RGB/depth/LiDAR not active in preview"),
        ]))
        self.spec_layout.addWidget(self._property_group("Export readiness", [
            ("Requirements", "captured from prompt"),
            ("Parts / BOM", "curated defaults until database match"),
            ("CAD / MJCF", "preview model available"),
            ("Scene set", "preview only"),
            ("Evaluation", "not run"),
            ("Export bundle", "not packaged"),
        ]))
        self._sync_scroll_height(self.spec_layout)

    def _inspector_banner(self, name: str, kind: str, chips: list[tuple[str, str]]) -> QFrame:
        f = QFrame(); f.setObjectName("inspectorBanner")
        f.setMinimumHeight(86)                          # grow (don't clip) when a long name wraps to 2 lines
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(6)
        title = QLabel(self._short(str(name), 56)); title.setObjectName("inspectorName"); title.setWordWrap(True)
        meta = QLabel(str(kind).upper()); meta.setObjectName("inspectorKind")
        lay.addWidget(self._hfw(title)); lay.addWidget(meta)
        row = QHBoxLayout(); row.setContentsMargins(0, 2, 0, 0); row.setSpacing(6)
        for text, state in chips:
            row.addWidget(self._status_chip(text, state))
        row.addStretch(1); lay.addLayout(row)
        return f

    def _status_chip(self, text: str, state: str = "metric") -> QLabel:
        lab = QLabel(str(text))
        lab.setObjectName("statusOk" if state == "ok" else ("statusWarn" if state == "warn" else "metricChip"))
        return lab

    def _property_group(self, title: str, rows: list[tuple[str, str]]) -> QFrame:
        f = QFrame(); f.setObjectName("propGroup")
        row_heights = [44 if len(str(value)) > 54 else 28 for _, value in rows]
        f.setMinimumHeight(43 + sum(row_heights))       # grow (don't clip) when a value wraps to 3+ lines
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 11); lay.setSpacing(8)
        h = QLabel(title.upper()); h.setObjectName("propTitle"); lay.addWidget(h)
        grid = QGridLayout(); grid.setContentsMargins(0, 0, 0, 0); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        for row, (key, value) in enumerate(rows):
            kk = QLabel(str(key)); kk.setObjectName("propKey"); kk.setMinimumWidth(106)
            vv = QLabel(str(value)); vv.setObjectName("propMono" if len(str(value)) < 28 else "propValue")
            vv.setMinimumHeight(18)
            vv.setWordWrap(True); vv.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(kk, row, 0, Qt.AlignTop)
            grid.addWidget(self._hfw(vv), row, 1)
        lay.addLayout(grid)
        return f

    @staticmethod
    def _task_route(prompt: str) -> str:
        low = (prompt or "").lower()
        skills = []
        for key, label in [("pick", "pick"), ("place", "place"), ("sort", "sort"),
                           ("carry", "carry"), ("walk", "locomotion"), ("run", "locomotion")]:
            if key in low and label not in skills:
                skills.append(label)
        return " -> ".join(skills) if skills else "compose -> simulate -> evaluate"

    @staticmethod
    def _short(text: str, limit: int = 80) -> str:
        text = str(text)
        return text if len(text) <= limit else text[:max(0, limit - 3)].rstrip() + "..."

    def _sync_scroll_height(self, layout):
        margins = layout.contentsMargins()
        count = layout.count()
        total = margins.top() + margins.bottom() + max(0, count - 1) * layout.spacing()
        for i in range(count):
            item = layout.itemAt(i)
            widget = item.widget() if item else None
            if widget is not None:
                total += max(widget.minimumHeight(), widget.sizeHint().height())
        parent = layout.parentWidget()
        if parent is not None:
            parent.setMinimumHeight(total)

    def _spec_block(self, title: str, rows: list, ok: bool = True, note: str | None = None):
        f = QFrame(); f.setObjectName("specCard"); lay = QVBoxLayout(f)
        lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(6)
        f.setMinimumHeight(52 + 42 * len(rows) + (28 if note else 0))
        hb = QHBoxLayout(); h = QLabel(title.upper()); h.setObjectName("h2"); hb.addWidget(h); hb.addStretch(1)
        badge = QLabel("VALID" if ok else "CHECK"); badge.setObjectName("badgeOk" if ok else "badgeBad")
        hb.addWidget(badge); lay.addLayout(hb)
        for k, val in rows:
            r = QGridLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setHorizontalSpacing(10)
            r.setColumnStretch(1, 1)
            kk = QLabel(str(k)); kk.setObjectName("specKey")
            vv = QLabel(str(val)); vv.setObjectName("specVal"); vv.setWordWrap(True)
            vv.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            r.addWidget(kk, 0, 0, Qt.AlignTop)
            r.addWidget(self._hfw(vv), 0, 1)
            lay.addLayout(r)
        if note:
            n = QLabel("Issue: " + note); n.setStyleSheet(f"color:{BAD}; font-size:11px;"); n.setWordWrap(True)
            lay.addWidget(self._hfw(n))
        self.spec_layout.addWidget(self._hfw(f))

    def _populate_tree(self, s: dict):
        self.tree.clear()
        root = QTreeWidgetItem([s.get("species") or s.get("robot_class", "robot"), "base"])
        self.tree.addTopLevelItem(root)
        for ln in s.get("links", []):
            QTreeWidgetItem(root, [str(ln), "link"])
        for e in (s.get("end_effectors", []) or []):
            QTreeWidgetItem(root, [str(e), "end-effector"])
        root.setExpanded(True)

    def _populate_tree_from_model(self, root_label: str):
        """Populate STRUCTURE from the model in the viewport  -  used for real / imported robots that
        have no composed gene to read link names from (otherwise the tab stays blank for them)."""
        self.tree.clear()
        bodies = self.viewport.body_tree()
        if not bodies:
            return
        root = QTreeWidgetItem([root_label, "base"])
        self.tree.addTopLevelItem(root)
        for name, njnt in bodies:
            role = f"{njnt} joint{'s' if njnt != 1 else ''}" if njnt else "fixed link"
            QTreeWidgetItem(root, [name, role])
        root.setExpanded(True)

    def _section_into(self, layout, title: str, rows: list):
        f = QFrame(); f.setObjectName("card"); lay = QVBoxLayout(f)
        lay.setContentsMargins(13, 11, 13, 11); lay.setSpacing(6)
        h = QLabel(title.upper()); h.setObjectName("h2"); lay.addWidget(h)
        for k, v in rows:
            r = QGridLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setHorizontalSpacing(10)
            r.setColumnStretch(1, 1)
            kk = QLabel(str(k)); kk.setObjectName("specKey"); kk.setMinimumWidth(150)  # align value columns
            vv = QLabel(str(v)); vv.setObjectName("specVal"); vv.setWordWrap(True)
            vv.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            r.addWidget(kk, 0, 0, Qt.AlignTop)
            r.addWidget(vv, 0, 1)
            lay.addLayout(r)
        layout.addWidget(f)

    def _withheld_card(self, why: str):
        """HONEST: we refuse to show a task score / 'proof of physics' unless a real simulated episode is
        actually playing in the viewport. No frozen viewport + a confident number ever again."""
        f = QFrame(); f.setObjectName("card"); lay = QVBoxLayout(f)
        lay.setContentsMargins(13, 11, 13, 11); lay.setSpacing(6)
        h = QLabel("RESULT WITHHELD"); h.setObjectName("h2"); h.setStyleSheet(f"color: {WARN};")
        lay.addWidget(h)
        msg = QLabel(why); msg.setWordWrap(True); msg.setStyleSheet(f"color: {MUTED};")
        lay.addWidget(msg)
        self.results_layout.addWidget(f)

    def _render_results(self, p: dict):
        self._clear_results()
        # TRUTH GATE: never show a task score / "proof of physics" unless a real episode is live in the viewport.
        if not self.viewport.has_live_episode():
            self._withheld_card("No simulation is playing in the viewport, so no task score is shown. A result "
                                "appears only when the robot actually ran its task on screen.")
            return
        auto, ev = p.get("autonomy", {}), p.get("evaluation", {})
        rate = ev.get("success_rate", auto.get("final_success_rate"))
        row = QHBoxLayout()
        row.addWidget(self._kpi("Task success", self._pct(rate), ACCENT if auto.get("succeeded") else WARN))
        row.addWidget(self._kpi("Objects placed", f"{ev.get('blocks_placed',' - ')}/{ev.get('blocks_total',' - ')}", TEXT))
        rw = QWidget(); rw.setLayout(row); self.results_layout.addWidget(rw)

        if auto.get("initial_success_rate") is not None and auto.get("final_success_rate") is not None \
                and auto["initial_success_rate"] != auto["final_success_rate"]:
            self._section("How it improved", [("As first built", self._pct(auto["initial_success_rate"])),
                                              ("After AI redesign", self._pct(auto["final_success_rate"]))])
        design = auto.get("converged_design") or {}
        if design:
            self._section("Body the AI chose",
                          [(k.replace("_", " "), f"{v:.1f}" if isinstance(v, (int, float)) else str(v))
                           for k, v in design.items()])
        notes = [n for n in (auto.get("notes") or []) if any(w in n.lower() for w in
                 ("redesign", "flywheel", "buildable", "lesson", "nearest", "reach"))]
        for n in notes[:4]:
            lab = self._faint(n); lab.setStyleSheet(f"color:{MUTED}; background:{SURFACE}; border:1px solid {BORDER};"
                                                    f"border-left:2px solid {ACCENT}; border-radius:6px; padding:8px;")
            self.results_layout.addWidget(lab)
        clusters = ev.get("failure_clusters") or []
        if clusters:
            self._section("Where it struggled", [(c.get("label", "?"), str(c.get("count", 0))) for c in clusters])
        compute = auto.get("compute") or {}
        if compute.get("physics_executed"):
            self._section("Proof of real physics",
                          [("MuJoCo steps", f"{compute.get('physics_steps', 0):,}"),
                           ("IK evals", f"{compute.get('ik_evaluations', 0):,}"),
                           ("Wall time", f"{compute.get('wall_time_seconds', 0)}s")])
        self._render_ledger(p.get("ledger") or {})

    def _render_ledger(self, led: dict):
        """The HONEST Product Readiness Ledger: each stage shows its real status (attained / placeholder /
        not_run / not_required), color-coded, and the bundle is only EXPORT-READY when every required stage is
        real. This is the truth surface — a scaffolded build cannot read as export-ready here."""
        stages = led.get("stages") or []
        if not stages:
            return
        color_for = {"attained": OK, "not_required": MUTED, "scaffolded": WARN, "placeholder": WARN,
                     "dry_run_only": WARN, "not_run": FAINT, "collision": BAD}
        f = QFrame(); f.setObjectName("card"); lay = QVBoxLayout(f)
        lay.setContentsMargins(13, 11, 13, 11); lay.setSpacing(6)
        safe = bool(led.get("safe_to_export"))
        h = QLabel("READINESS LEDGER  ·  " + ("EXPORT-READY" if safe else "EXPORT-BLOCKED (honest)"))
        h.setObjectName("h2"); h.setStyleSheet(f"color: {OK if safe else WARN};")
        lay.addWidget(h)
        for st in stages:
            status = str(st.get("status", "?"))
            r = QGridLayout(); r.setContentsMargins(0, 0, 0, 0); r.setHorizontalSpacing(10); r.setColumnStretch(1, 1)
            kk = QLabel(str(st.get("stage", "")).replace("_", " ")); kk.setStyleSheet(f"color: {MUTED};")
            kk.setMinimumWidth(150)
            vv = QLabel(status.replace("_", " ")); vv.setStyleSheet(
                f"color: {color_for.get(status, TEXT)}; font-family: Consolas, monospace;")
            r.addWidget(kk, 0, 0, Qt.AlignTop); r.addWidget(vv, 0, 1)
            lay.addLayout(r)
        self.results_layout.addWidget(f)

    def _kpi(self, label: str, value: str, color: str) -> QFrame:
        f = QFrame(); f.setObjectName("card"); f.setMinimumHeight(76)
        lay = QVBoxLayout(f); lay.setContentsMargins(15, 13, 15, 13); lay.setSpacing(4)
        lv = QLabel(label.upper()); lv.setObjectName("kpiLabel")
        vv = QLabel(value); vv.setObjectName("kpi"); vv.setStyleSheet(f"color: {color}; font-size: 34px; font-weight: 800;")
        lay.addWidget(lv); lay.addWidget(vv); lay.addStretch(1); return f

    def _section(self, title: str, rows: list):
        f = QFrame(); f.setObjectName("card"); lay = QVBoxLayout(f); lay.setContentsMargins(13, 11, 13, 11); lay.setSpacing(6)
        h = QLabel(title.upper()); h.setObjectName("h2"); lay.addWidget(h)
        for k, v in rows:
            r = QGridLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setHorizontalSpacing(10)
            r.setColumnStretch(1, 1)
            kk = QLabel(str(k)); kk.setStyleSheet(f"color: {MUTED};")
            vv = QLabel(str(v)); vv.setStyleSheet(f"color: {TEXT}; font-family: 'JetBrains Mono', Consolas, monospace;")
            vv.setWordWrap(True)
            vv.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            r.addWidget(kk, 0, 0, Qt.AlignTop)
            r.addWidget(vv, 0, 1)
            lay.addLayout(r)
        self.results_layout.addWidget(f)

    # ---- project summary (no HTTP; read the package files directly) ----
    def _project_summary(self) -> dict:
        if not (self.project_dir / "robot" / "robot_genome.json").exists():
            return {"built": False}
        out = {"built": True}
        for key, rel in [("autonomy", "reports/autonomy_report.json"),
                         ("evaluation", "reports/physics_evaluation_report.json"),
                         ("ledger", "reports/product_readiness_ledger.json")]:
            pp = self.project_dir / rel
            if pp.exists():
                try:
                    out[key] = json.loads(pp.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
        # gene_evaluation_report.json is what gene_build writes for a built legged/manipulator robot — read it so the
        # KPI reflects the actual (honest, cadence-gated) build result instead of showing blank when no separate
        # physics_evaluation_report exists.
        gene_rep = self.project_dir / "reports" / "gene_evaluation_report.json"
        if "evaluation" not in out and gene_rep.exists():
            try:
                g = json.loads(gene_rep.read_text(encoding="utf-8"))
                out["evaluation"] = {"success_rate": g.get("success_rate"), "task_type": g.get("task_type"),
                                     "status": g.get("status"), "forward_m": g.get("forward_m"),
                                     "blocks_placed": g.get("blocks_placed"), "blocks_total": g.get("blocks_total")}
            except Exception:  # noqa: BLE001
                pass
        nav = self.project_dir / "reports" / "navigation_evaluation_report.json"
        if "evaluation" not in out and nav.exists():
            try:
                n = json.loads(nav.read_text(encoding="utf-8"))
                out["evaluation"] = {"success_rate": n.get("success_rate"), "blocks_placed": n.get("reached"),
                                     "blocks_total": n.get("total_episodes"), "failure_clusters": n.get("failure_clusters", [])}
            except Exception:  # noqa: BLE001
                pass
        return out

    @staticmethod
    def _pct(x) -> str:
        return " - " if x is None else f"{round(x * 100)}%"

    @staticmethod
    def _design_label(raw: str) -> str:
        """Present a banked design's stored prompt as a clean name, not a raw command string: strip a leading
        command verb ('build ...'/'compose ...') and capitalise, so 'build So i want a humanoid...' reads as
        'So i want a humanoid...' truncated."""
        s = (raw or "").strip()
        for verb in ("build ", "compose ", "design ", "make ", "create "):
            if s.lower().startswith(verb):
                s = s[len(verb):].lstrip(); break
        if s:
            s = s[0].upper() + s[1:]
        return (s[:34] + "…") if len(s) > 35 else s


def _register_app_fonts():
    for font_path in [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/consolab.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/Menlo.ttc"),
    ]:
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))


def _apply_theme(app: QApplication):
    _register_app_fonts()
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont(SANS, 9))
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG)); pal.setColor(QPalette.Base, QColor(SURFACE))
    pal.setColor(QPalette.Text, QColor(TEXT)); pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(SURFACE2)); pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT)); pal.setColor(QPalette.HighlightedText, QColor("#03201c"))
    app.setPalette(pal)
    app.setStyleSheet(_qss())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Virturoid Desktop (standalone).")
    ap.add_argument("--workspace", default=str(Path("build") / "desktop"))
    args = ap.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    _apply_theme(app)
    win = MainWindow(Path(args.workspace))
    win.showMaximized()        # fit the screen so the right-edge cluster/inspector never clips off-screen

    # CRASH GUARD: a Python exception raised on the GUI thread (e.g. inside a signal slot that touches MuJoCo)
    # used to tear down the whole process silently ("the app randomly closed"). Keep the app alive and surface
    # the error instead; faulthandler dumps a real traceback if a native (MuJoCo/GL) fault ever occurs.
    import faulthandler
    import traceback
    faulthandler.enable()

    def _excepthook(et, e, tb):
        traceback.print_exception(et, e, tb)
        try:
            win.statusBar().showMessage(f"Internal error (handled): {e}", 8000)
            if hasattr(win, "_chat_ai"):
                win._chat_ai(f"<span style='color:{BAD}'>Something errored, but I kept the app open.</span> {e}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = _excepthook
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
