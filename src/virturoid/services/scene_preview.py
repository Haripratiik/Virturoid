from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet


def write_scene_previews(scene_sets: list[SceneSet], output_dir: Path) -> list[Path]:
    """Write dependency-free top-down SVG previews for generated scene sets."""
    written: list[Path] = []
    preview_dir = output_dir / "simulation" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for scene_set in scene_sets:
        if not scene_set.scenes:
            continue
        path = preview_dir / f"{scene_set.purpose}_scene.svg"
        path.write_text(scene_to_svg(scene_set.scenes[0], title=f"{scene_set.purpose.title()} Scene Preview"), encoding="utf-8")
        written.append(path)

    index_path = preview_dir / "index.html"
    index_path.write_text(_preview_index(scene_sets), encoding="utf-8")
    written.append(index_path)
    return written


def scene_to_svg(scene: SceneGraph, title: str = "Scene Preview") -> str:
    width = 720
    height = 520
    scale = 420.0
    origin_x = 120
    origin_y = 260
    objects = "\n".join(_object_svg(item, origin_x, origin_y, scale) for item in scene.objects)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f7f8fa" />
  <text x="24" y="34" font-family="Arial" font-size="22" fill="#1f2933">{escape(title)}</text>
  <text x="24" y="58" font-family="Arial" font-size="13" fill="#5c6670">Scene: {escape(scene.name)}</text>
  <rect x="60" y="110" width="560" height="320" rx="4" fill="#ddd8cc" stroke="#9b9488" />
  <text x="68" y="128" font-family="Arial" font-size="12" fill="#5a5148">table/workspace</text>
  <circle cx="{origin_x}" cy="{origin_y}" r="18" fill="#222831" />
  <text x="{origin_x - 28}" y="{origin_y + 40}" font-family="Arial" font-size="12" fill="#222831">robot base</text>
  {objects}
</svg>
"""


def _object_svg(item: SceneObject, origin_x: int, origin_y: int, scale: float) -> str:
    x, y, _, _, _, _ = item.pose_xyz_rpy
    cx = origin_x + x * scale
    cy = origin_y - y * scale
    color = _color_for_object(item)
    label = escape(item.name)
    if item.object_type in {"container", "conveyor"}:
        width = 54 if item.object_type == "container" else 120
        height = 42 if item.object_type == "container" else 34
        return (
            f'  <rect x="{cx - width / 2:.1f}" y="{cy - height / 2:.1f}" width="{width}" height="{height}" '
            f'rx="3" fill="{color}" stroke="#333" stroke-width="1.5" />\n'
            f'  <text x="{cx - width / 2:.1f}" y="{cy + height / 2 + 14:.1f}" font-family="Arial" font-size="11" fill="#222">{label}</text>'
        )
    return (
        f'  <rect x="{cx - 14:.1f}" y="{cy - 14:.1f}" width="28" height="28" rx="3" fill="{color}" '
        f'stroke="#333" stroke-width="1.5" />\n'
        f'  <text x="{cx - 22:.1f}" y="{cy + 30:.1f}" font-family="Arial" font-size="11" fill="#222">{label}</text>'
    )


def _color_for_object(item: SceneObject) -> str:
    material = (item.material or "").lower()
    name = item.name.lower()
    if "red" in material or "red" in name:
        return "#d7443e"
    if "blue" in material or "blue" in name:
        return "#3867d6"
    if "cardboard" in material or "box" in item.object_type:
        return "#b88753"
    if item.object_type == "conveyor":
        return "#353b48"
    if item.object_type == "container":
        return "#8f98a3"
    return "#7f8c8d"


def _preview_index(scene_sets: list[SceneSet]) -> str:
    links = []
    for scene_set in scene_sets:
        if scene_set.scenes:
            href = f"{scene_set.purpose}_scene.svg"
            links.append(f'<li><a href="{escape(href)}">{escape(scene_set.purpose.title())} scene preview</a></li>')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Virturoid Scene Previews</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #20242a; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>Virturoid Scene Previews</h1>
  <ul>{''.join(links)}</ul>
</body>
</html>
"""
