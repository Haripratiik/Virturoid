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
- Fits a walking gait to each new body by search (CEM) against an un-gameable verdict; optional GPU training (MJX PPO) behind a CPU↔GPU parity gate when a CUDA box is attached.
- Learns a contact grasp for arms, which can sort objects by color.
- Trains under domain randomization (actuator gain, joint stiffness, sensor noise, and pushes) for sim-to-real robustness.

**Run tasks**
- Proposes a verifiable task from the prompt and checks it against the robot's morphology.
- Generates the scenes to test it, then runs the real skill: pick and place, sort, navigate, or locomote.
- Measures the outcome instead of assuming it.

**Close the sim-to-real gap on a robot you own**
- Writes the bench experiment to run on your actual hardware: a short, safe, information-rich command sequence, one joint at a time, with every amplitude bounded by that joint's own declared limits and every frequency bounded by the datasheet torque and no-load speed of the motor its bill of materials sized. A joint it cannot move safely is reported as such rather than commanded anyway.
- Measures the gap from the log you send back — **per joint, in radians, milliseconds and newton-metres, never a single fidelity score**. It replays your commands through the simulator to compare trajectories, then subtracts the simulator's own inverse dynamics from your measured torque and regresses the remainder on the model's sensitivity, so it names *which* joints and *which* parameters are responsible. Actuation delay is identified separately by re-simulating the closed loop across a delay grid.
- Fits each joint's viscous damping, reflected inertia and dry friction with a **confidence interval, not a point estimate**, and applies only what survives two refusals: parameters the experiment could not actually load are reported as unidentified rather than guessed, and a fit that does not measurably improve how the simulator tracks your log is withheld from the model entirely. Every applied calibration is reversible in one call and carries the prior it replaced.
- No hardware yet? The same journey runs against a deliberately perturbed copy of the model, and labels every result as a simulation rather than a measurement — including refusing to raise the actuator-fidelity level. Honest scope: this validates the pipeline and the estimator, not the physics. Both sides are MuJoCo there, so MuJoCo's own modelling error cancels, and that is exactly the error a real log exists to expose. **No number we publish has been validated against a physical robot; the first hardware log needs a design partner.**

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
- Hands off to NVIDIA Isaac Sim / Isaac Lab: exports an OpenUSD physics articulation (transcribed from the exact model Virturoid simulates, then re-read and round-tripped through OpenUSD to confirm it loads cleanly) plus a ready-to-edit Isaac Lab `ArticulationCfg` with real per-joint motor limits, a standalone spawn script, and, for legged robots, a velocity-tracking locomotion environment that subclasses Isaac Lab's own task. Virturoid designs and pre-screens the robot; your Isaac pipeline does the high-fidelity training and sim-to-real.

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

The Robot Genome compiles to a MuJoCo model and runs in real physics. What ships today is a **scripted gait that is tuned per body by search** — and the tuning is real learning, not a lookup.

A structural wave-gait engine drives any leg count, and a **CEM search** fits that gait's parameters to each new body against an un-gameable reward (forward travel counts only when the robot also stayed upright, kept a real step cadence, and survived the episode). Leg **stiffness** turned out to be the decisive dimension: on a spindly body the same gait is rejected as a *slide* at `kp=32` and passes as a **credible walk** at `kp=250`. An OpenAI-ES trainer for morphology-agnostic policies (an attention network reading one token per joint) also runs on CPU.

**GPU training is wired but optional.** With a CUDA box attached, MJX runs PPO over thousands of parallel robots behind an enforced CPU↔GPU parity gate, and a policy is only banked if it earns a credible verdict on the CPU deploy path. PPO converges in simulation; closing the last of the sim-to-deploy gap into a banked neural walk is the open frontier, so **no learned neural policy ships as the default controller** — the honest headline is *tuned, verified gaits that compound as reusable assets*.

Rather than learn a gait from a dead stop, the policy learns a **residual on top of a rhythmic gait prior**, using position control toward a default stance. A policy that starts from pure noise tends to collapse into a lunge, but giving it a rhythm to refine produces a robot that takes real steps and stays upright. Arms learn a **contact grasp** the same way. Training can run under **domain randomization** so a controller is robust to the gap between simulation and real hardware.

### 4. Running a task

A robot is judged by whether it can do the job, not just whether it stands up. From the prompt, Virturoid **proposes a verifiable task**, checks it against the robot's morphology so the task fits the body, **generates a scene specific to that task** — sorting bins, a stacking target, a push goal, a lift shelf, a navigation course, or a maze sized to the robot — and runs the matching **real skill**: pick and place, sort, navigate, or locomote. The result is measured, and a build that fails its task is reported as such.

![Each task generates its own scene, sized to the robot: sort, stack, push, lift, navigate, maze](assets/scene_generation.png)

### 5. Two AI loops

- **Body designer.** A language model turns the prompt into the anatomy graph.
- **Reward critic.** A second language model reads a diagnosis of how a gait turned out, such as step cadence, balance, and foot clearance, and rewrites the training reward weights to push toward a cleaner result. This applies the language-to-rewards idea to gait quality.

### 6. The flywheel

Every verified gait and skill is banked into a **morphology vector space** and a linked **project memory**. When you ask for a new robot, Virturoid finds its nearest neighbors and **warm-starts from their tuned parameters** instead of searching from scratch — a brand-new quadruped recalls real banked gait parameters, not the shipped defaults.

What is measured today is **asset compounding**: the library of verified, recallable gaits grows with use, and warm-started bodies start from a working region instead of zero. The harness that would prove *capability* compounding — a held-out success curve against shuffled-label and other controls — is built and unit-tested, but has not yet been run at corpus scale, so it is stated as designed, not proven.

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
git clone https://github.com/Haripratiik/Virturoid.git && cd Virturoid
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[sim]"         # physics + CAD: everything below runs on this
```

**`.[sim]` is the install to start with.** It is what the demo, the studio, the verdicts and the CAD export
actually use. Measured on a clean venv (Windows, Python 3.13): **223 s to install, 696 MB on disk** — most of it
OpenCascade, which is what makes the CAD real. `.[all]` also pulls PyTorch, JAX and PySide6 (several GB) and is
only worth it once you want GPU training or the native desktop window.

**`pip install -e .` is what makes `import virturoid` work.** The package lives in `src/`, so a clone alone is
not importable — without the editable install, `import virturoid` fails from every directory, including the
repo root. The core install needs only NumPy (69 MB venv) and gives you an importable library plus the
`virturoid`, `virturoid-build`, `virturoid-import` and `virturoid-mvp` commands; the extras add the engines.

Not ready to install? The two entry-point scripts — `scripts/run_mvp_demo.py` and `scripts/run_ui.py` — put
`src/` on `sys.path` themselves, so those two run straight from a clone (you still need MuJoCo). Everything
else, including your own code, expects the install or `PYTHONPATH=src` (Windows: `set PYTHONPATH=src`).

| Extra | Pulls in | For |
|---|---|---|
| *(none)* | NumPy | schemas, planning, memory, the CLIs |
| `sim` | MuJoCo, Pillow, build123d | **start here** — physics, verdicts, renders, STEP/STL CAD |
| `web` | FastAPI, uvicorn | the FastAPI web app (`run_ui.py` itself needs nothing extra) |
| `desktop` | PySide6, pywebview | Studio as a native window instead of a browser tab |
| `rl` | JAX, PyTorch | GPU/MJX training and the morphology-aware policies |
| `isaac` | usd-core | the OpenUSD / Isaac Lab hand-off |
| `llm` | openai, anthropic, requests | the LLM design backends (offline is the default) |
| `all` | all of the above | full local development |

## Usage

### Start here — two robots, no API key, no GPU (**allow 4–10 minutes**)

```bash
python scripts/run_mvp_demo.py --mini      # -> build/demo/index.html : robots built from text, each with its verdict
```

**Budget the time before you start it.** This used to be documented as "~1 minute" and it is not: two cold runs
on the development machine measured **212 s and 336 s**, and an evaluator on a different machine measured
**579 s**. Almost all of it is one stage — the quadruped's `create_robot` took 202 s, 321 s and 574 s in those
three runs, while the arm that follows it took **7 s**. That is not a hang and it is
not overhead — `create_robot` grounds the body to its real mass and then *searches* for a gait that fits that
particular body, against a verdict that can see it fall. Fitting a body's controller to the body is the
expensive, honest version of that step; the cheap version is scoring your robot at some other robot's
hand-tuned operating point, which is what the search exists to stop.

So the run talks while it works. Every stage announces itself, reports what it cost, and prints a
`... still running` line every 15 seconds with a clock in the left margin:

```
[0:00]    [1/2] a quadruped robot dog
[0:00]       create_robot ... (compose -> ground the mass -> fit an operating point to THIS body ...)
[0:15]         ... still running: create_robot (15s)
[3:22]       done create_robot in 202.2s
[3:23]       done verify_robot (quick) in 0.5s
[3:23]    -> CREDIBLE WALK  (203.8s for this robot)
```

The output is a self-contained gallery: every robot is composed from a prompt, simulated in MuJoCo, and
labelled with the verdict it actually earned — including the ones that fail. Nothing here calls an LLM, so it
runs on a fresh clone with no keys configured — and the page says so at the top, naming the design path it
resolved and stamping every card with the one that produced that body. Run `--llm` (or set
`VIRTUROID_LLM_BACKEND=openai` + `OPENAI_API_KEY`) to have a language model author the anatomy instead of the
offline compositor; the physics, verdicts, and exports are identical either way.

The full run (`python scripts/run_mvp_demo.py`, **measured 703 s / 11m43s** on the same machine) builds seven
bodies — four of them legged, each with a flywheel learning pass — and adds the measurement that matters most: a
**same-family comparison** — three quadruped-animal prompts, three measurably different bodies, each verified on
its own, with a `SUBSTITUTED` column that says so if the walkability gate ever replaces one with a shared
template. `--no-compare` skips it; `--no-learn` skips the learning passes.

### Virturoid Studio (the app)

Studio is the real frontend — a React + Vite desktop/web app (source in [`frontend/`](frontend/), served by the
Python backend at `/studio/`). **The built bundle is committed, so a clone needs no npm step:**

```bash
python scripts/run_ui.py --ui studio --web --port 8765    # -> http://127.0.0.1:8765/studio/
python scripts/run_ui.py --ui studio                      # or as a native desktop window (needs the desktop extra)
```

With `--ui studio`, `http://127.0.0.1:8765/` **redirects to `/studio/`** and the older lightweight build console
keeps its own address at `/legacy`. (Without the flag it is the other way round: the console is at `/`, Studio
still at `/studio/`.) Opening the root and landing on the legacy console — whose viewport reads *"Unknown
package."* until you pick one — was the single easiest way to conclude the app was broken while Studio was
running one path segment away.

**What you should see on a fresh clone:** four demo robots (`arm_sort`, `dog_walk`, `hexapod_walk`, `humanoid`)
in the Robot Library — they are tracked in git under `build/ui_verify/`, and the server falls back to them when
your own build root is still empty. The status column is deliberately unflattering: most of them read
`UNVERIFIED` until you verify them. If the library *is* empty it now says which directory it scanned and how to
fill it, instead of showing a blank grid.

If you change the frontend, rebuild it with `cd frontend && npm install && npm run build`, or develop against
it with hot reload (below).

Describe a robot to the build assistant and it builds it in the live 3D viewport. Switch the viewport to **Episode** to replay the trained motion, open **Memory** for the cross-robot species tree, and **Analysis** for evaluation detail. To develop the frontend with hot reload, run the backend as above and `cd frontend && npm install && npm run dev` (Vite serves `http://localhost:5173/studio/` and proxies the API to the backend).

The original lightweight Build Console is also reachable directly with `python -m virturoid.ui_server [--web --port 8765]`.

The native desktop window needs the `desktop` extra (`pip install -e ".[desktop]"`, which pulls in `pywebview`);
without it the launcher says so and falls back to the browser. (`site/` is a separate Astro marketing site and is
not part of the app.)

### Command line

The studio is the product, but the whole engine is scriptable.

```bash
# Compose a robot, co-design it into a working body, and evaluate it on its task in real physics
python -m virturoid.compose --prompt "warehouse arm to move 2 kg boxes with 0.9 m reach" --co-design --evaluate --build build/arm

# One command, from prompt to a working simulated robot
python -m virturoid.autobuild --prompt "a tabletop arm that sorts red and blue blocks into matching bins" --output build/sorter

# Build, train a controller, and export a bundle plus a runnable ROS 2 package
python -m virturoid.build --train --prompt "a tabletop arm that sorts blocks" --output build/arm_train

# Import an existing robot model (URDF or MJCF) and recover an editable robot from it
python -m virturoid.import_robot path/to/robot.urdf --save-gene build/imported/gene.json

# Browse a gallery of robots built from text, each verified in real physics
python scripts/run_mvp_demo.py                      # writes a self-contained build/demo/index.html
python scripts/run_mvp_demo.py --llm                # ...designed by the configured LLM backend instead of offline

# End-to-end: ingest a folder of an existing robot (model + BOM + CAD + description + control script) and improve it
python scripts/demo_ingest_customer.py
```

Real-world URDFs (a Unitree Go2, a Franka arm) often don't load in a strict simulator as-published; the importer
runs a deterministic repair pass (normalizes materials, resolves mesh paths) and reports every change, so your
robot comes in with its own link names and structure preserved rather than replaced by a generic body.

## Connect your own AI agent (MCP)

Virturoid is agent-first: it runs on **your** LLM subscription, not ours. Point Claude Code / Claude Desktop
(or any MCP client) at the built-in server and it can author, edit, verify, train, export, and **ingest** robots
through one tool surface — with zero tokens billed to Virturoid.

```bash
# start the server (stdio JSON-RPC; nothing is billed to us)
python -m virturoid.mcp_server

# or register it with Claude Code
claude mcp add virturoid -- python -m virturoid.mcp_server
```

The server advertises a lean workflow menu (`create_robot`, `edit_robot`, `verify_robot`, `export_held`,
`ingest_project`, …); `ingest_project` is the gateway for bringing in an existing robot/BOM/policy/dataset.

Advanced authoring tools round out the loop: `train_reward` runs the closed reward-as-code loop (author an objective, optimize it, and bank the verified reward pattern to the flywheel), `generate_fusion` compiles an EKF/AHRS/odometry sensor-fusion config from the robot's real BOM sensors, and `generate_control_scripts` emits the obs-assembler + state-machine + safety-filter + watchdog control stack.

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
| `ANTHROPIC_API_KEY` | Your key when the backend is `claude` |
| `VIRTUROID_CLAUDE_MODEL` | Model name for the Claude backend |
| `VIRTUROID_LOCAL_LLM_URL` / `VIRTUROID_LOCAL_LLM_MODEL` | Endpoint + model when the backend is `local` (Ollama / vLLM / any OpenAI-compatible server) |
| `VIRTUROID_GPU_SSH` | SSH target of a GPU box for training, for example `user@host` |

**Studio's chat assistant is configured separately.** The variables above choose the model that authors robot
*anatomy* in the build pipeline; the free-form chat box in Studio is its own swappable layer with its own knobs:

| Variable | Purpose | Default |
|---|---|---|
| `VIRTUROID_ASSISTANT_PROVIDER` | Chat backend for the Studio assistant. Currently `ollama` is the only implemented provider | `ollama` |
| `VIRTUROID_ASSISTANT_MODEL` | Model tag the assistant asks that provider for | `llama3.2` |
| `OLLAMA_HOST` | Base URL of the Ollama runtime, when the provider is `ollama` | `http://127.0.0.1:11434` |

You do **not** need any of these to build robots: describe a robot in the chat box and the request is dispatched
to the deterministic build pipeline whether or not a chat model is running. They only enable free-form
conversation. Setting `VIRTUROID_LLM_BACKEND=openai` does not configure this assistant, and vice versa.

Bring your own subscription: set `VIRTUROID_LLM_BACKEND` to `openai`, `claude`, or `local` and supply the
matching key/endpoint — the keys are yours and never leave your machine. Everything also runs fully offline
(`off`), which is the default.

## Roadmap

- **Learned humanoid locomotion.** Bring the learning stack from quadrupeds to a balancing biped.
- **Faster, command-conditioned gaits.** Steer a trained policy by target speed and direction.
- **Onboard perception.** Range sensing and vision so robots can act in unknown environments.
- **A hardware log.** Measuring the sim-to-real gap ships today (see *Close the sim-to-real gap on a robot you own*), but every number it has produced so far came from a simulated stand-in. Running the bench experiment on a physical robot, and carrying a trained controller onto it, needs a design partner.

## License

Released under the MIT License. See [LICENSE](LICENSE).
