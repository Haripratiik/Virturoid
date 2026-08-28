"""Pure-stdlib safety gate (no ROS/numpy): clamp commanded joint-position targets to joint
limits + a per-step rate (velocity) limit. The LAST thing before a real motor -- it prevents a policy from
driving a joint past its mechanical stop or slewing faster than the safety budget. The node calls clamp() on
every command before publishing to ros2_control. Mirrors services/sim_ros_bridge.SafetyFilter."""


class SafetyFilter:
    def __init__(self, lower, upper, vel_limit=8.0):
        self.lower, self.upper, self.vel_limit = list(lower), list(upper), float(vel_limit)

    def clamp(self, target, q, dt):
        """Return (clamped_targets, n_violations); a violation = the raw command had to be altered."""
        out, violations = [], 0
        step = max(1e-6, self.vel_limit * float(dt))
        for i, t in enumerate(target):
            c = min(self.upper[i], max(self.lower[i], float(t)))
            c = min(q[i] + step, max(q[i] - step, c))
            if abs(c - float(t)) > 1e-6:
                violations += 1
            out.append(c)
        return out, violations
