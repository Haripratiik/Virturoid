# Virturoid

**An AI-native robot creation engine.** Describe a robot in plain language, and Virturoid designs its body, sizes a real bill of materials, generates fabrication-ready CAD, simulates it in real physics, trains its controller, and runs it on a task. Every robot it builds makes the next one faster to create.

You write something like *"a four-legged robot that walks"* or *"a tabletop arm that sorts blocks"*, and the system composes an original body for it, chooses real motors and sensors to build it, runs it inside a physics simulator, teaches it to move through reinforcement learning, and checks that it can actually do the job.

It runs as a native desktop studio with a live 3D viewport, and the whole engine is also scriptable from the command line. There are no hand-coded robot templates. One general pipeline takes any morphology from prompt to trained controller.

![Virturoid pipeline: from a prompt to a trained, buildable robot](assets/architecture.svg)

Every body below was generated from a one-line prompt by the same pipeline — no per-species templates:

![Robots generated from prompts: a manipulator, quadruped, hexapod, mobile base, humanoid, and octopod](assets/robot_gallery.png)

## Table of contents

- [What makes it different](#what-makes-it-different)
- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Built with](#built-with)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [License](#license)

## What makes it different

- **It runs the whole loop, not one slice.** Most tools stop at generating a shape, or at a physics demo, or at a controller. Virturoid goes from a sentence to a buildable robot, to a trained controller, to a robot that completes a task, to an export bundle.
- **One policy controls any body.** A single morphology-agnostic policy drives a quadruped, a hexapod, or an arm. Learning is not rewritten for each robot, and what one body learns can carry to the next.
- **Every robot is original.** Bodies are generated from the prompt through one general compiler. Nothing is retrieved from a catalog of stock models or stitched together from existing robot parts.
- **It compounds.** The flywheel banks every trained robot and warm-starts the next similar one, so the system gets faster and cheaper the more it is used. That growing library, not any single model, is the asset.
- **AI designs it and AI critiques it.** One language model designs the body; a second reads how training went and rewrites the reward. The system improves its own training signal.
- **It is honest by construction.** A robot is marked ready to export only when real artifacts back every stage, and a gait is scored by real foot contact and balance.

## What it does

**Design**
- Generates an original robot body from a natural-language prompt.
- Realizes any morphology through one general anatomy compiler, with no per-species templates.
- Runs fully offline with a deterministic composer when no language model is configured.
- Co-designs the body, physics-tuning it into a working robot before it is built.

**Build**
- Sizes every joint to a real off-the-shelf actuator from a component catalog.
- Assembles a complete bill of materials: actuators, sensors, compute, and power.
- Produces real parametric B-rep CAD with build123d, exported to STEP and STL.
- Adapts materials to the task, such as heavier steel for load or lighter carbon for agility.

**Simulate and learn**
- Compiles each robot to a MuJoCo model and trains it in real physics.
- Learns locomotion with one morphology-agnostic attention policy, trained with PPO on the GPU through MJX.
- Learns a contact grasp for arms, which can sort objects by color.
- Trains under domain randomization (actuator gain, joint stiffness, sensor noise, and pushes) for sim-to-real robustness.

**Run tasks**
- Proposes a verifiable task from the prompt and checks it against the robot's morphology.
- Generates the scenes to test it, then runs the real skill: pick and place, sort, navigate, or locomote.
- Measures the outcome instead of assuming it.

**Reuse and organize**
- Banks every trained body and skill into a morphology vector space and a linked project memory.
- Warm-starts each new robot from the most similar past work instead of training from scratch.
- Places every design into a self-organizing species tree.

**Use it**
- A native desktop studio with a live MuJoCo viewport, or a full command-line interface.
- Edits a built robot in plain language — make it taller, give it carbon-fiber legs, or make it carry 10 kg — and re-engineers the body for it, sizing bigger motors and updating the bill of materials, then re-verifying.
- Ingests an existing robot project: drop a folder with a URDF or MJCF model, a bill of materials, CAD meshes, and a plain-English description, and one agent parses all of it into a single editable, simulate-able robot — even when the referenced meshes are missing.
- Runs and improves your own controller: hand it your control script or policy and it executes it in real physics, then tunes it into a better gait.
- Exports a controller bundle, a runnable ROS 2 package, and browsable reports.

## How it works

Virturoid runs as an explicit pipeline, the one in the diagram above. Every stage produces a real, inspectable artifact, so the path from prompt to trained robot stays transparent instead of hidden in a black box. Each stage, in order:

### 1. From a prompt to a body

You describe the robot in natural language. A language model interprets the request into an **anatomy graph**: its limbs, segments, and joints, their proportions, and how they connect. When no API key is set, a deterministic composer builds the same kind of graph offline.

A single **general anatomy compiler** then turns that graph into real 3D geometry. The same compiler handles a dog, a hexapod, or a robot arm, so there are no per-species templates to maintain. The output is a **Robot Genome**, the canonical specification that every later stage reads. An optional **co-design** step physics-tunes the body before it is built, so it is shaped to actually perform its task.

### 2. Real, buildable hardware

A design is only useful if you could actually build it, so Virturoid grounds every robot in real parts.

- **Actuators.** Each joint is sized to a real off-the-shelf motor from a component catalog, matched to the torque and speed the joint needs.
- **Bill of materials.** The system assembles a complete parts list covering actuators, sensors, compute, and power.
- **CAD.** Geometry is real parametric CAD built with build123d on OpenCascade, exported as B-rep STEP and STL files, with materials chosen to fit the task.

### 3. Learning to move

The Robot Genome compiles to a MuJoCo model and runs in real physics. Control is **learned, not scripted**.

Locomotion uses one **morphology-agnostic policy**: an attention network that reads one token per joint, so the same architecture controls a quadruped, a hexapod, or an arm, and what it learns on one body can transfer to another. Training runs **PPO on the GPU** through MJX, stepping thousands of simulated robots in parallel.

Rather than learn a gait from a dead stop, the policy learns a **residual on top of a rhythmic gait prior**, using position control toward a default stance. A policy that starts from pure noise tends to collapse into a lunge, but giving it a rhythm to refine produces a robot that takes real steps and stays upright. Arms learn a **contact grasp** the same way. Training can run under **domain randomization** so a controller is robust to the gap between simulation and real hardware.

### 4. Running a task

A robot is judged by whether it can do the job, not just whether it stands up. From the prompt, Virturoid **proposes a verifiable task**, checks it against the robot's morphology so the task fits the body, **generates a scene specific to that task** — sorting bins, a stacking target, a push goal, a lift shelf, a navigation course, or a maze sized to the robot — and runs the matching **real skill**: pick and place, sort, navigate, or locomote. The result is measured, and a build that fails its task is reported as such.

![Each task generates its own scene, sized to the robot: sort, stack, push, lift, navigate, maze](assets/scene_generation.png)

### 5. Two AI loops

- **Body designer.** A language model turns the prompt into the anatomy graph.
- **Reward critic.** A second language model reads a diagnosis of how a gait turned out, such as step cadence, balance, and foot clearance, and rewrites the training reward weights to push toward a cleaner result. This applies the language-to-rewards idea to gait quality.

### 6. The flywheel

Every trained body and skill is banked into a **morphology vector space** and a linked **project memory**. When you ask for a new robot, Virturoid finds its nearest neighbors and **warm-starts from their learned weights** instead of training from scratch.

The effect compounds. On a block-sorting task, a second build of a robot reused prior work and cut the number of simulated candidates from 173 to 23, roughly a 7.5x saving. The longer the system runs, the cheaper each new robot becomes, and that growing library is the core asset.

### 7. The readiness gate

Each build is checked stage by stage against the artifacts actually on disk: real CAD geometry, a real physics pass, and measured task outcomes. A design is marked ready to export only when the evidence is present, which keeps the studio honest about what a given robot can really do.

### 8. Bring your own robot

Virturoid is not only a generator; it is a simulation home for robots you already have. Drop a project folder — a URDF or MJCF model, a bill of materials, CAD meshes, and a plain-English description like *"aluminum chassis, carbon-fiber legs, carries a 5 kg payload"* — and one ingestion agent parses all of it into a single editable robot. It imports the model (recovering the kinematic structure even when the referenced meshes are missing), reads the description into typed materials and payload and applies them, and folds in the parts list. From there the same tools that build a robot amend and improve it: ask it to carry more and it re-sizes the motors and updates the bill of materials. You can also hand it your own control script or policy — it runs the controller in real physics, then warm-starts a search from your parameters to tune it into a better gait, and keeps your controller if it cannot honestly beat it.

## Project structure

```
virturoid/
├── src/virturoid/
│   ├── schemas/         Typed data models: Robot Genome, BOM, CAD, scenes, tasks, training, readiness
│   ├── services/        The engine: anatomy design and compiler, CAD, BOM, physics, training, tasks, flywheel
│   ├── ui_server.py     Build Console: the product UI (native window or browser), serves webui/
│   ├── build.py         Generic builder CLI with morphology-aware routing
│   ├── autobuild.py     One-command autonomous build, prompt to working robot
│   ├── compose.py       Compose and co-design a body from building blocks
│   └── import_robot.py  Import an existing MJCF or URDF robot and train it
├── webui/               Build Console front end: 3D viewport, episode playback, outliner, memory
├── scripts/             Training, evaluation, and utility scripts
├── tests/               Test suite
├── pyproject.toml       Package, extras, and console entry points
└── README.md
```

Good entry points into the engine: `anatomy_designer.py` and `anatomy_compiler.py` (prompt to geometry), `bom_builder.py` and `component_catalog.py` (real parts), `cad_geometry.py` (B-rep CAD), `morph_policy.py`, `learn_locomotion.py`, and `gpu_trainer.py` (learned control), `task_proposer.py`, `task_verifier.py`, and `task_executor.py` (running a task), `gait_critic.py` (the reward critic), `design_flywheel.py` and `skill_flywheel.py` (reuse), and `readiness_ledger.py` (the export gate).

## Built with

- **Python 3.10+**
- **MuJoCo** and **MJX** for physics and GPU-parallel simulation
- **JAX** for GPU training, with **PyTorch** for supporting models
- **build123d** on OpenCascade for parametric B-rep CAD
- **Three.js** front end served by a built-in Python server, run as a native window or in the browser
- **NumPy** as the only required runtime dependency
- Pluggable language-model backends: OpenAI, Claude, a local model through Ollama or vLLM, or a fully offline composer

## Installation

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[all]"         # full install, or ".[desktop,sim]" for just the studio
```

The core package needs only NumPy, so `pip install -e .` gives you an importable library and runnable CLIs without the heavy engines. The extras (`sim`, `cad`, `rl`, `desktop`, `web`, `all`) pull in MuJoCo, CAD, and the learning stack as needed.

## Usage

### Build Console (the studio)

```bash
python -m virturoid.ui_server                       # native window
python -m virturoid.ui_server --web --port 8765     # or in the browser at http://127.0.0.1:8765
```

Describe a robot to the build assistant and it builds it in the live 3D viewport. Switch the viewport to **Episode** to replay the trained motion, open **Memory** for the cross-robot species tree, and **Analysis** for evaluation detail.

### Command line

The studio is the product, but the whole engine is scriptable.

```bash
# Compose a robot, co-design it into a working body, and evaluate it on its task in real physics
python -m virturoid.compose --prompt "warehouse arm to move 2 kg boxes with 0.9 m reach" --co-design --evaluate --build build/arm

# One command, from prompt to a working simulated robot
python -m virturoid.autobuild --prompt "a tabletop arm that sorts red and blue blocks into matching bins" --output build/sorter

# Build, train a controller, and export a bundle plus a runnable ROS 2 package
python -m virturoid.build --train --prompt "a tabletop arm that sorts blocks" --output build/arm_train

# Import an existing robot model and learn a controller for it
python -m virturoid.import_robot --mjcf-file path/to/robot.xml --output build/imported

# Browse a gallery of robots built from text, each verified in real physics
python scripts/run_mvp_demo.py                      # writes a self-contained build/demo/index.html

# End-to-end: ingest a folder of an existing robot (model + BOM + CAD + description + control script) and improve it
python scripts/demo_ingest_customer.py
```

`--co-design` physics-tunes a freshly composed body before building, `--evaluate` scores it on its morphology-matched task, and `--benchmark` scores it across a difficulty suite. Every build writes a complete package: the Robot Genome, the compiled MuJoCo model, generated task and scene sets, the bill of materials, parametric and B-rep CAD, training artifacts, a controller bundle, and browsable reports. Open `reports/index.html` in any package to explore everything it generated.

## Configuration

Virturoid runs fully offline by default. To enable language-model design, copy the template and set your backend:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `VIRTUROID_LLM_BACKEND` | Which language-model backend to use: `off`, `openai`, `claude`, or `local` |
| `OPENAI_API_KEY` | Your key when the backend is `openai` |
| `VIRTUROID_OPENAI_MODEL` | Model name for the OpenAI backend |
| `VIRTUROID_GPU_SSH` | SSH target of a GPU box for training, for example `user@host` |

## Roadmap

- **Learned humanoid locomotion.** Bring the learning stack from quadrupeds to a balancing biped.
- **Faster, command-conditioned gaits.** Steer a trained policy by target speed and direction.
- **Onboard perception.** Range sensing and vision so robots can act in unknown environments.
- **Sim-to-real transfer.** Carry trained controllers onto physical hardware.

## License

Released under the MIT License. See [LICENSE](LICENSE).
