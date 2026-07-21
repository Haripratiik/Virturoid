// The plain-language glossary. Every piece of robotics jargon shown in the UI
// should be registered here so <Term> can explain it inline. Definitions are
// written for someone who has never touched robotics; experts just ignore them.

export interface GlossaryEntry {
  term: string;
  short: string;
}

export const GLOSSARY = {
  dof: {
    term: "DOF — degrees of freedom",
    short: "How many independent ways the robot can move. A human arm has 7; more DOF means more dexterity but harder control.",
  },
  urdf: {
    term: "URDF",
    short: "A standard file format that describes a robot's body: its parts, how they connect, and how its joints move. Used by ROS and most robotics tools.",
  },
  mujoco: {
    term: "MuJoCo",
    short: "The physics simulator used here — it computes gravity, contact and friction so the robot behaves like it would in the real world.",
  },
  joint: {
    term: "Joint",
    short: "A connection between two body parts that allows movement, like an elbow or a wheel axle. Motors drive joints.",
  },
  revolute: {
    term: "Revolute joint",
    short: "A joint that rotates around one axis, like a door hinge or your elbow.",
  },
  prismatic: {
    term: "Prismatic joint",
    short: "A joint that slides in a straight line, like a drawer.",
  },
  link: {
    term: "Link",
    short: "One rigid body part of the robot — an upper arm, a chassis plate, a wheel. Links are connected by joints.",
  },
  actuator: {
    term: "Actuator",
    short: "The motor that powers a joint. Each moving joint needs one.",
  },
  effort: {
    term: "Effort limit",
    short: "The maximum turning force (torque) the joint's motor can apply, in newton-metres.",
  },
  torque: {
    term: "Torque",
    short: "Turning force, measured in newton-metres (Nm). More torque lets a joint lift heavier things.",
  },
  bom: {
    term: "BOM — bill of materials",
    short: "The shopping list for building this robot physically: every part, its quantity, weight and price.",
  },
  endEffector: {
    term: "End effector",
    short: "The robot's 'hand' — the tool at the end of the arm that grips or interacts with objects.",
  },
  payload: {
    term: "Payload",
    short: "The heaviest object the robot is designed to pick up or carry.",
  },
  reach: {
    term: "Reach",
    short: "How far the robot's arm can extend from its base.",
  },
  scene: {
    term: "Scene",
    short: "A simulated test environment — a table, objects, bins — where the robot's behavior is exercised.",
  },
  episode: {
    term: "Episode",
    short: "One recorded attempt at the task inside the physics simulator, played back frame by frame like a video.",
  },
  baseline: {
    term: "Baseline scene",
    short: "The standard, expected version of the test environment.",
  },
  variation: {
    term: "Variation scene",
    short: "The same test with things shuffled — different object positions, colors or counts — to prove the robot isn't memorizing one setup.",
  },
  edgeCase: {
    term: "Edge-case scene",
    short: "A deliberately hard or unusual setup (clutter, extremes) designed to find where the robot breaks.",
  },
  successRate: {
    term: "Success rate",
    short: "The share of simulated attempts where the robot completed its task. 100% across many varied scenes is strong evidence it works.",
  },
  training: {
    term: "Training (RL)",
    short: "Reinforcement learning: the robot tries its task thousands of times in simulation, gets scored, and gradually improves its own controller.",
  },
  policy: {
    term: "Policy",
    short: "The robot's learned 'brain' — the function that decides which motor commands to send based on what it senses.",
  },
  curriculum: {
    term: "Curriculum",
    short: "The training syllabus: easier versions of the task first, harder ones later — like grade school for robots.",
  },
  honestyGate: {
    term: "Honesty gate",
    short: "A check that separates what has been physically proven in simulation from what is only estimated or claimed. Robots ship only after passing.",
  },
  grounding: {
    term: "Grounding",
    short: "Evidence linking a claim to a real artifact — a simulation run, a physics log — instead of just the AI asserting it.",
  },
  scorecard: {
    term: "Honesty scorecard",
    short: "The per-robot report of which capabilities are proven, which are scaffolded (partially real) and which are placeholders.",
  },
  placeholder: {
    term: "Placeholder",
    short: "A stand-in value that has NOT been verified — flagged so it is never mistaken for a real result.",
  },
  genome: {
    term: "Robot genome",
    short: "The single structured file that fully describes this robot — body, joints, sensors, actuators — from which everything else is generated.",
  },
  sim2real: {
    term: "Sim-to-real",
    short: "The gap between simulation and the physical world. Varied scenes and honest physics narrow it.",
  },
  ik: {
    term: "Inverse kinematics",
    short: "The math that answers: 'what joint angles put the hand exactly there?'",
  },
  export: {
    term: "Export bundle",
    short: "The zip that leaves the studio: robot description, meshes, scenes, training results and the honesty scorecard — ready for ROS or a real build.",
  },
  localModel: {
    term: "Local AI model",
    short: "The language model powering the agent runs on THIS machine (via Ollama). Nothing leaves your computer; builds still work without it.",
  },
  quat: {
    term: "Quaternion",
    short: "A four-number way to store 3D rotation without the glitches of angles. You rarely need to touch these directly.",
  },
} as const satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof GLOSSARY;
