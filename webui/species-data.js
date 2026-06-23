// Curated robotics knowledge base for the Memory ("second brain") workspace.
// The DEFAULT view is the ROBOT SPECIES TREE: robot classes and the species the
// pipeline actually produces. Each species note carries the task-specific BUILD +
// TRAIN tips the agent has learned. A "Full graph" toggle adds the wider task /
// skill / training / concept web. Frontend-owned reference content, seeded with
// the real pipeline species.

// Restrained palette: lime accent for classes, near-white for species,
// steel for capabilities (task/skill), slate for method/meta (training/concept).
export const GROUPS = {
  root: { label: "Root", color: "#e8c06b" },
  class: { label: "Robot class", color: "#b8ff3a" },
  species: { label: "Species", color: "#e3ebf0" },
  build: { label: "Built robot", color: "#7fdfff" },
  task: { label: "Task", color: "#7fb6c2" },
  skill: { label: "Skill", color: "#7fb6c2" },
  training: { label: "Training", color: "#94a2ae" },
  concept: { label: "Concept", color: "#94a2ae" },
};

export const NODES = [
  // ---- Root (trunk of the species tree) ----
  { id: "robots", label: "Robots", group: "root", tags: ["species-tree"],
    note: "The robot **species tree**. Every robot class and the species the pipeline has produced branch from here — click a species to read its build + training tips, or use **Full graph** for the wider task / skill / training web." },

  // ---- Robot classes ----
  { id: "manipulator", label: "Manipulator", group: "class", tags: ["arm", "fixed-base"],
    note: "**Manipulators** are fixed-base arms. Arm prompts route here. Key levers: **degrees of freedom** (more DOF = larger workspace, harder control), **reach (m)**, and **payload (kg)**." },
  { id: "mobile_base", label: "Mobile Base", group: "class", tags: ["wheeled", "navigation"],
    note: "**Mobile bases** drive through an environment. Navigation / delivery / maze prompts route here. Differential drive is the default species." },
  { id: "quadruped", label: "Quadruped", group: "class", tags: ["legged", "4-leg"],
    note: "**Quadrupeds** are four-legged walkers. Walk / locomotion prompts route here. The pipeline gives 12 actuated joints (3 per leg) and a learned trot gait." },
  { id: "hexapod", label: "Hexapod", group: "class", tags: ["legged", "6-leg"],
    note: "**Hexapods** are six-legged walkers — more legs, more static stability, a slower gait than a quadruped." },
  { id: "octopod", label: "Octopod", group: "class", tags: ["legged", "8-leg"],
    note: "**Octopods** are eight-legged walkers — maximal static stability, the heaviest body to actuate." },
  { id: "humanoid", label: "Humanoid", group: "class", tags: ["legged", "biped"],
    note: "**Humanoids** are bipeds — the hardest class to control because balance is inherently unstable. Generated from humanoid prompts." },

  // ---- Species (real pipeline species), each with task-specific BUILD + TRAIN tips ----
  { id: "fixed_arm.three_dof.tabletop", label: "fixed_arm.three_dof.tabletop", group: "species", tags: ["3-DOF", "tabletop"],
    note: "Three-DOF tabletop arm — the workhorse for tabletop manipulation.\n\n**Build tips**\n- 3 revolute joints + a parallel-jaw gripper; reach 0.4–0.6 m, payload 0.5–1.5 kg.\n- Pair with an RGB-D camera for object detection; keep links light to protect dynamics.\n\n**Train tips by task**\n- Sort / pick-place: a learned contact-grasp residual with domain randomization on friction; add a color-classification step for sorting.\n- Stack: reward place precision + a stable-for-2s bonus; release timing matters more than reach.\n- Lift: a heavier payload needs higher PD gains; target an elevated shelf, not a bin.\n- Push: no grasp — reward puck-to-zone distance and steer around obstacles." },
  { id: "differential_drive.two_wheel.compact", label: "differential_drive.two_wheel.compact", group: "species", tags: ["diff-drive"],
    note: "Compact two-wheel differential-drive base for indoor navigation.\n\n**Build tips**\n- Small footprint for tight scenes; LiDAR for mapping-heavy tasks, RGB-D for close range.\n\n**Train tips by task**\n- Navigate: reward goal-reach + obstacle clearance; cap speed near obstacles.\n- Maze: the course is sized to the robot's footprint — reward progress through corridors and penalize wall contact.\n- Deliver: navigation plus a docking bonus at the drop zone." },
  { id: "quadruped.composed.12dof", label: "quadruped.composed.12dof", group: "species", tags: ["trot", "12-DOF"],
    note: "Composed quadruped, 12 actuated joints (3 per leg) — the headline learned walker.\n\n**Build tips**\n- 3 DOF per leg (hip + knee); foot friction and balanced trunk mass set stability.\n- Per-joint PD gains scale with effective inertia, so the recipe generalizes across leg lengths.\n\n**Train tips**\n- Learn a residual on top of a trot-CPG prior with position control to a default stance — a policy from a dead stop collapses into a lunge.\n- Reward velocity tracking + an alive bonus; terminate on fall; add foot-air-time for cleaner stepping.\n- Train under domain randomization (actuator gain, mass, pushes) for sim-to-real. Real result: forward 0.45–0.83 m, upright, cadence ~13." },
  { id: "hexapod.composed.18dof", label: "hexapod.composed.18dof", group: "species", tags: ["tripod", "18-DOF"],
    note: "Composed hexapod, 18 joints (3 per leg). More legs → more static stability, a slower gait.\n\n**Build tips**\n- Wide stance; the body rarely topples, so balance is easier than a quadruped.\n\n**Train tips**\n- The same CPG-prior-plus-residual recipe as the quadruped; an alternating tripod gait (two sets of three legs) is the stable default." },
  { id: "octopod.composed.24dof", label: "octopod.composed.24dof", group: "species", tags: ["8-leg"],
    note: "Composed octopod, eight legs. Maximal static stability, the heaviest body to actuate.\n\n**Build tips**\n- Redundant legs make standing trivial but raise actuator count and mass.\n\n**Train tips**\n- Reuse the legged recipe; redundancy means a single mis-stepping leg rarely tips the body, so the alive bonus dominates early training." },
  { id: "humanoid.anthropometric", label: "humanoid.anthropometric", group: "species", tags: ["biped"],
    note: "Anthropometric biped — torso, head, two arms, two legs — from the same general anatomy compiler.\n\n**Build tips**\n- Human-like proportions and foot size; no special-case template.\n\n**Train tips (honest)**\n- A bipedal CPG + balance prior keeps it upright briefly: it **generates and balances**, but does **not walk yet**.\n- A real walk needs GPU-PPO with careful reward shaping and a curriculum — bipedal locomotion is the hard open problem here (on the roadmap). Don't report a walk it can't do." },

  // ---- Tasks (the wider 'Full graph' web) ----
  { id: "pick_place", label: "Pick & Place", group: "task", tags: ["grasp"],
    note: "Move objects from A to B. The most reliable first task. Success depends on **grasp** stability and **perception** of object pose." },
  { id: "sorting", label: "Sorting", group: "task", tags: ["classification"],
    note: "Pick objects and route them to bins by class (e.g. color). Adds a **classification** step on top of pick/place." },
  { id: "navigation", label: "Navigation", group: "task", tags: ["path"],
    note: "Drive to goal poses without collision. Needs **SLAM/mapping** or a known map, plus **motion planning**." },
  { id: "delivery", label: "Delivery", group: "task", tags: ["logistics"],
    note: "Navigation + payload transport between stations. Combine a mobile base with reliable docking." },
  { id: "locomotion", label: "Locomotion", group: "task", tags: ["walk"],
    note: "Move the whole body forward while staying upright. Scored HONESTLY by forward distance, cadence (real foot lifts), and upright fraction — not raw displacement, which a slide or topple can fake." },

  // ---- Skills ----
  { id: "grasping", label: "Grasping", group: "skill", tags: ["gripper"],
    note: "Forming and maintaining a stable grip. Watch out for slip; in sim, contact parameters dominate. Train with **domain randomization** on friction." },
  { id: "perception", label: "Perception", group: "skill", tags: ["vision"],
    note: "Turning sensor data into object/world estimates. **RGB-D** gives depth + color; **LiDAR** gives accurate range. Synthetic observations are generated per scene." },
  { id: "motion_planning", label: "Motion Planning", group: "skill", tags: ["trajectory"],
    note: "Computing collision-free trajectories. For arms, joint-space planning; for bases, path planning on a costmap." },
  { id: "slam", label: "SLAM / Mapping", group: "skill", tags: ["localization"],
    note: "Simultaneous localization and mapping for mobile bases. LiDAR-based SLAM is the most robust starting point." },

  // ---- Training ----
  { id: "domain_randomization", label: "Domain Randomization", group: "training", tags: ["sim2real"],
    note: "Randomize physics (friction, mass), visuals, and noise during training so the policy generalizes. The single highest-leverage trick for **sim2real**." },
  { id: "reward_shaping", label: "Reward Shaping", group: "training", tags: ["reward"],
    note: "Design dense, well-scaled rewards. Add shaping terms (distance-to-goal, grasp bonus) but keep the true objective dominant to avoid reward hacking." },
  { id: "curriculum", label: "Curriculum", group: "training", tags: ["staged"],
    note: "Start easy, increase difficulty as the policy improves (fewer distractors → more; closer goals → farther). Speeds convergence dramatically." },
  { id: "sim2real", label: "Sim2Real", group: "training", tags: ["transfer"],
    note: "Closing the gap between simulation and hardware. Driven by domain randomization, accurate contact models, and conservative actuation limits." },
  { id: "ppo", label: "PPO", group: "training", tags: ["rl"],
    note: "Proximal Policy Optimization — a stable on-policy RL algorithm, a strong default for continuous control. Tune entropy and clip range first." },

  // ---- Concepts ----
  { id: "urdf", label: "URDF", group: "concept", tags: ["model"],
    note: "Unified Robot Description Format — the robot's links, joints, and visual meshes. The viewport renders this directly." },
  { id: "mjcf", label: "MJCF / MuJoCo", group: "concept", tags: ["sim"],
    note: "MuJoCo scene XML describing the world, objects, and physics. Compiled scenes drive evaluation and training." },
  { id: "readiness", label: "Readiness Gates", group: "concept", tags: ["quality"],
    note: "A scored checklist of whether a package meets the MVP bar (valid model, scenes, contract). Optional stages (CV, training) are skipped, not failed, when a build doesn't include them." },
  { id: "validation", label: "Validation", group: "concept", tags: ["quality"],
    note: "File-level checks that every expected artifact exists and parses. Failing validation usually means a generation step didn't complete." },
];

export const LINKS = [
  // root -> classes (the trunk of the species tree)
  ["robots", "manipulator"],
  ["robots", "mobile_base"],
  ["robots", "quadruped"],
  ["robots", "hexapod"],
  ["robots", "octopod"],
  ["robots", "humanoid"],

  // class -> species (the species tree)
  ["manipulator", "fixed_arm.three_dof.tabletop"],
  ["mobile_base", "differential_drive.two_wheel.compact"],
  ["quadruped", "quadruped.composed.12dof"],
  ["hexapod", "hexapod.composed.18dof"],
  ["octopod", "octopod.composed.24dof"],
  ["humanoid", "humanoid.anthropometric"],

  // species -> task (full graph)
  ["fixed_arm.three_dof.tabletop", "pick_place"],
  ["fixed_arm.three_dof.tabletop", "sorting"],
  ["differential_drive.two_wheel.compact", "navigation"],
  ["differential_drive.two_wheel.compact", "delivery"],
  ["quadruped.composed.12dof", "locomotion"],
  ["hexapod.composed.18dof", "locomotion"],
  ["octopod.composed.24dof", "locomotion"],
  ["humanoid.anthropometric", "locomotion"],

  // task -> skill / training
  ["pick_place", "grasping"],
  ["sorting", "grasping"],
  ["sorting", "perception"],
  ["pick_place", "motion_planning"],
  ["navigation", "slam"],
  ["navigation", "motion_planning"],
  ["delivery", "navigation"],
  ["locomotion", "reward_shaping"],
  ["grasping", "domain_randomization"],
  ["perception", "domain_randomization"],
  ["grasping", "reward_shaping"],
  ["navigation", "reward_shaping"],
  ["locomotion", "ppo"],
  ["domain_randomization", "sim2real"],
  ["reward_shaping", "ppo"],
  ["curriculum", "ppo"],
  ["ppo", "sim2real"],

  // concepts
  ["manipulator", "urdf"],
  ["mobile_base", "urdf"],
  ["urdf", "mjcf"],
  ["mjcf", "readiness"],
  ["readiness", "validation"],
  ["perception", "mjcf"],
];
