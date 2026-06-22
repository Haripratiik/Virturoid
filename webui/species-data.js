// Curated robotics knowledge base for the Memory ("second brain") workspace.
// Nodes are concepts/species/skills; links connect related ideas. Notes hold
// practical building + training tips. This is frontend-owned reference content,
// seeded with the real species the pipeline produces.

// Restrained palette: lime accent for classes, near-white for species,
// steel for capabilities (task/skill), slate for method/meta (training/concept).
export const GROUPS = {
  class: { label: "Robot class", color: "#b8ff3a" },
  species: { label: "Species", color: "#e3ebf0" },
  task: { label: "Task", color: "#7fb6c2" },
  skill: { label: "Skill", color: "#7fb6c2" },
  training: { label: "Training", color: "#94a2ae" },
  concept: { label: "Concept", color: "#94a2ae" },
};

export const NODES = [
  // Classes
  { id: "manipulator", label: "Manipulator", group: "class", tags: ["arm", "fixed-base"],
    note: "**Manipulators** are fixed-base robot arms. Virturoid routes arm prompts here. Key levers: **degrees of freedom** (more DOF = larger reachable workspace but harder control), **reach (m)**, and **payload (kg)**.\n\nTips:\n- Start with 3-DOF tabletop for pick/place; add DOF only if the task needs orientation control.\n- Keep payload realistic \u2014 over-spec'd payload inflates link mass and hurts dynamics." },
  { id: "mobile_base", label: "Mobile Base", group: "class", tags: ["wheeled", "navigation"],
    note: "**Mobile bases** drive through an environment. Routed from navigation/delivery prompts. Differential drive is the default species.\n\nTips:\n- Mobile bases usually have no visual meshes yet \u2014 the viewport shows primitive fallbacks.\n- Sensor choice matters: LiDAR for mapping/SLAM, RGB-D for close-range perception." },

  // Species (real pipeline species)
  { id: "fixed_arm.three_dof.tabletop", label: "fixed_arm.three_dof.tabletop", group: "species", tags: ["3-DOF", "tabletop"],
    note: "Three-DOF tabletop arm. The workhorse species for **pick/place** and **sorting** demos.\n\nBuild tips:\n- Reach ~0.4\u20130.6 m, payload ~0.5\u20131.5 kg is a stable starting envelope.\n- Pair with an **RGB-D camera** for block detection." },
  { id: "differential_drive.two_wheel.compact", label: "differential_drive.two_wheel.compact", group: "species", tags: ["diff-drive"],
    note: "Compact two-wheel differential-drive base for indoor navigation and delivery.\n\nBuild tips:\n- Use **LiDAR** for mapping-heavy tasks.\n- Keep the footprint small for tight indoor scenes." },

  // Tasks
  { id: "pick_place", label: "Pick & Place", group: "task", tags: ["grasp"],
    note: "Move objects from A to B. The most reliable first task. Success depends on **grasp** stability and **perception** of object pose." },
  { id: "sorting", label: "Sorting", group: "task", tags: ["classification"],
    note: "Pick objects and route them to bins by class (e.g. color). Adds a **classification** step on top of pick/place." },
  { id: "navigation", label: "Navigation", group: "task", tags: ["path"],
    note: "Drive to goal poses without collision. Needs **SLAM/mapping** or a known map, plus **motion planning**." },
  { id: "delivery", label: "Delivery", group: "task", tags: ["logistics"],
    note: "Navigation + payload transport between stations. Combine a mobile base with reliable docking." },

  // Skills
  { id: "grasping", label: "Grasping", group: "skill", tags: ["gripper"],
    note: "Forming and maintaining a stable grip. Watch out for slip; in sim, contact parameters dominate. Train with **domain randomization** on friction." },
  { id: "perception", label: "Perception", group: "skill", tags: ["vision"],
    note: "Turning sensor data into object/world estimates. **RGB-D** gives depth + color; **LiDAR** gives accurate range. Synthetic observations are generated per scene." },
  { id: "motion_planning", label: "Motion Planning", group: "skill", tags: ["trajectory"],
    note: "Computing collision-free trajectories. For arms, joint-space planning; for bases, path planning on a costmap." },
  { id: "slam", label: "SLAM / Mapping", group: "skill", tags: ["localization"],
    note: "Simultaneous localization and mapping for mobile bases. LiDAR-based SLAM is the most robust starting point." },

  // Training
  { id: "domain_randomization", label: "Domain Randomization", group: "training", tags: ["sim2real"],
    note: "Randomize physics (friction, mass), visuals, and noise during training so the policy generalizes. The single highest-leverage trick for **sim2real**." },
  { id: "reward_shaping", label: "Reward Shaping", group: "training", tags: ["reward"],
    note: "Design dense, well-scaled rewards. Add shaping terms (distance-to-goal, grasp bonus) but keep the true objective dominant to avoid reward hacking." },
  { id: "curriculum", label: "Curriculum", group: "training", tags: ["staged"],
    note: "Start easy, increase difficulty as the policy improves (fewer distractors \u2192 more; closer goals \u2192 farther). Speeds convergence dramatically." },
  { id: "sim2real", label: "Sim2Real", group: "training", tags: ["transfer"],
    note: "Closing the gap between simulation and hardware. Driven by domain randomization, accurate contact models, and conservative actuation limits." },
  { id: "ppo", label: "PPO", group: "training", tags: ["rl"],
    note: "Proximal Policy Optimization \u2014 a stable on-policy RL algorithm, a strong default for continuous control. Tune entropy and clip range first." },

  // Concepts
  { id: "urdf", label: "URDF", group: "concept", tags: ["model"],
    note: "Unified Robot Description Format \u2014 the robot's links, joints, and visual meshes. The viewport renders this directly." },
  { id: "mjcf", label: "MJCF / MuJoCo", group: "concept", tags: ["sim"],
    note: "MuJoCo scene XML describing the world, objects, and physics. Compiled scenes drive evaluation and training." },
  { id: "readiness", label: "Readiness Gates", group: "concept", tags: ["quality"],
    note: "A scored checklist of whether a package meets the MVP bar (valid URDF, scenes, contract, etc.). Aim to clear all **required** gates." },
  { id: "validation", label: "Validation", group: "concept", tags: ["quality"],
    note: "File-level checks that every expected artifact exists and parses. Failing validation usually means a generation step didn't complete." },
];

export const LINKS = [
  ["manipulator", "fixed_arm.three_dof.tabletop"],
  ["mobile_base", "differential_drive.two_wheel.compact"],
  ["fixed_arm.three_dof.tabletop", "pick_place"],
  ["fixed_arm.three_dof.tabletop", "sorting"],
  ["differential_drive.two_wheel.compact", "navigation"],
  ["differential_drive.two_wheel.compact", "delivery"],
  ["pick_place", "grasping"],
  ["sorting", "grasping"],
  ["sorting", "perception"],
  ["pick_place", "motion_planning"],
  ["navigation", "slam"],
  ["navigation", "motion_planning"],
  ["delivery", "navigation"],
  ["grasping", "domain_randomization"],
  ["perception", "domain_randomization"],
  ["grasping", "reward_shaping"],
  ["navigation", "reward_shaping"],
  ["domain_randomization", "sim2real"],
  ["reward_shaping", "ppo"],
  ["curriculum", "ppo"],
  ["ppo", "sim2real"],
  ["manipulator", "urdf"],
  ["mobile_base", "urdf"],
  ["urdf", "mjcf"],
  ["mjcf", "readiness"],
  ["readiness", "validation"],
  ["perception", "mjcf"],
];
