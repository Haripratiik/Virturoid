"""Robot Parser agent (plan §8, Phase C): URDF/MJCF text → Robot Genome draft.

A high-volume specialist agent (routed to the local Nemotron-class model) that
reads a robot description (URDF or MJCF, possibly messy or non-standard) and
proposes a structured Robot Genome draft — links, joints with parent/child wiring,
and end effectors — plus warnings about anything it had to infer. The proposal is
validated as a real kinematic graph: every joint must connect known links, the
graph must be a single connected tree with exactly one root and no cycles, and the
end effectors must exist. Invalid drafts are fed back to the model (self-repair).

With no LLM backend it returns None and the caller falls back to a deterministic
URDF parser — offline-safe. ``to_robot_genome`` converts a validated draft into a
``RobotGenome`` that passes ``RobotGenome.validate()``.
"""

from __future__ import annotations

from virturoid.schemas.robot import RobotGenome, RobotJoint

_JOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "joint_type": {"type": "string"},
        "parent_link": {"type": "string"},
        "child_link": {"type": "string"},
    },
    "required": ["name", "joint_type", "parent_link", "child_link"],
    "additionalProperties": False,
}

PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "links": {"type": "array", "items": {"type": "string"}},
        "joints": {"type": "array", "items": _JOINT_SCHEMA},
        "end_effectors": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "links", "joints", "end_effectors"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's robot parser. Read the robot description (URDF or MJCF) and output "
    "ONLY JSON describing its kinematic graph: name, links (every body/link name), joints "
    "(each {name, joint_type, parent_link, child_link}), end_effectors (tip link names), and "
    "warnings for anything you inferred or that looked malformed. The result must be a valid "
    "kinematic tree: each joint connects two known links, there is exactly one root link (never "
    "a child), no cycles, and every link is connected. A validator checks this and may ask you "
    "to correct it."
)


def parse_robot_description(description: str, source_format: str, llm, max_repairs: int = 2) -> dict | None:
    """Parse a URDF/MJCF string into a genome draft; validate + self-repair. None if no backend."""
    if llm is None:
        return None

    user = (
        f"Format: {source_format}. Robot description:\n```\n{description}\n```\n"
        "Return JSON: {name, links, joints[{name,joint_type,parent_link,child_link}], "
        "end_effectors, warnings}."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, PARSER_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"draft": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        issues = _validate(raw)
        if not issues:
            return {"draft": raw, "backend": getattr(llm, "name", "?"), "attempts": attempts,
                    "valid": True, "warnings": raw.get("warnings", [])}
        last_issues = issues
        user = f"{user}\n\nYour previous draft {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"draft": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _validate(raw: dict) -> list[str]:
    issues: list[str] = []
    links = raw.get("links") or []
    link_set = set(links)
    if not links:
        issues.append("links must be non-empty")
    if len(link_set) != len(links):
        issues.append("link names must be unique")

    joints = raw.get("joints") or []
    children: set[str] = set()
    parent_of: dict[str, str] = {}
    for i, j in enumerate(joints):
        if not isinstance(j, dict):
            issues.append(f"joints[{i}] malformed")
            continue
        p, c = j.get("parent_link"), j.get("child_link")
        if p not in link_set:
            issues.append(f"joints[{i}] parent_link {p!r} is not a known link")
        if c not in link_set:
            issues.append(f"joints[{i}] child_link {c!r} is not a known link")
        if c in children:
            issues.append(f"link {c!r} is the child of more than one joint (not a tree)")
        children.add(c)
        parent_of[c] = p

    # Kinematic-tree checks: exactly one root, no cycles, all links connected.
    if link_set:
        roots = [l for l in link_set if l not in children]
        if len(roots) != 1:
            issues.append(f"expected exactly one root link, found {len(roots)}: {sorted(roots)}")
        if not _is_acyclic(parent_of):
            issues.append("kinematic graph has a cycle")
        elif len(roots) == 1 and joints:
            reachable = _reachable_from_root(roots[0], joints, link_set)
            unreachable = link_set - reachable
            if unreachable:
                issues.append(f"links not connected to the root: {sorted(unreachable)}")

    ees = raw.get("end_effectors") or []
    if not ees:
        issues.append("end_effectors must be non-empty")
    for ee in ees:
        if ee not in link_set:
            issues.append(f"end_effector {ee!r} is not a known link")
    return issues


def _is_acyclic(parent_of: dict[str, str]) -> bool:
    for start in parent_of:
        seen = set()
        node = start
        while node in parent_of:
            if node in seen:
                return False
            seen.add(node)
            node = parent_of[node]
    return True


def _reachable_from_root(root: str, joints: list[dict], link_set: set[str]) -> set[str]:
    adj: dict[str, list[str]] = {l: [] for l in link_set}
    for j in joints:
        p, c = j.get("parent_link"), j.get("child_link")
        if p in adj and c in adj:
            adj[p].append(c)
    seen = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def to_robot_genome(result: dict, species: str = "imported_arm") -> RobotGenome:
    """Convert a validated parser draft into a RobotGenome that passes validate()."""
    d = result["draft"]
    return RobotGenome(
        id=f"genome_{d['name']}",
        name=d["name"],
        species=species,
        links=list(d["links"]),
        joints=[
            RobotJoint(
                name=j["name"], joint_type=j["joint_type"],
                parent_link=j["parent_link"], child_link=j["child_link"],
            )
            for j in d["joints"]
        ],
        end_effectors=list(d["end_effectors"]),
    )
