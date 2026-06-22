"""Render the real desktop MainWindow to a PNG so we can see/iterate on the UI without a live display."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")   # don't fire GPU/LLM workers during the grab
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"          # predictable 1x capture
os.environ["QT_SCALE_FACTOR"] = "1"

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402

from virturoid.desktop import MainWindow, _apply_theme  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "build/_ui/current.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

app = QApplication.instance() or QApplication(sys.argv)
_apply_theme(app)
import tempfile
from pathlib import Path
w = MainWindow(Path(tempfile.mkdtemp(prefix="vui_")))
_wd = int(sys.argv[2]) if len(sys.argv) > 2 else 1560
_ht = int(sys.argv[3]) if len(sys.argv) > 3 else 980
w.resize(_wd, _ht)
w.show()
if len(sys.argv) > 4:                                  # optional page: 0=BUILD 1=MEMORY 2=SETTINGS
    w.main_stack.setCurrentIndex(int(sys.argv[4]))
for _ in range(8):
    app.processEvents()


def _grab():
    w.grab().save(out)
    print("saved", out)
    app.quit()


QTimer.singleShot(1200, _grab)
app.exec()
