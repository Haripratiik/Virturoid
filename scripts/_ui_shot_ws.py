"""Render the desktop MainWindow against an EXPLICIT workspace (so the flywheel/memory page shows real
banked data) to a PNG. Usage: _ui_shot_ws.py <out.png> <workspace> <w> <h> [page 0=build/1=memory/2=settings]."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")   # don't fire GPU/LLM workers during the grab
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"

from pathlib import Path  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from virturoid.desktop import MainWindow, _apply_theme  # noqa: E402

out = sys.argv[1]
ws = Path(sys.argv[2])
wd = int(sys.argv[3]) if len(sys.argv) > 3 else 1360
ht = int(sys.argv[4]) if len(sys.argv) > 4 else 720
page = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

app = QApplication.instance() or QApplication(sys.argv)
_apply_theme(app)
w = MainWindow(ws)
w.resize(wd, ht)
w.show()
w.main_stack.setCurrentIndex(page)
for _ in range(8):
    app.processEvents()


def _grab():
    w.grab().save(out)
    print("saved", out)
    app.quit()


QTimer.singleShot(1400, _grab)
app.exec()
