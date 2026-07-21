# THE LINE — a 3D robot arm that carries the build through the page

**Status**: research + planning only. No code changes accompany this document.
**Owner split**: user builds the Blender asset (§6 is the build sheet) · agent builds all code (§5, §7–§11).

---

## 0. The concept, stated precisely

A **3D robot arm model** is pinned in the viewport. In front of it runs a production line with
**one workpiece — "the part"**. As the viewer scrolls, the arm **picks the part up and moves it
along the line**, station by station, in exact sync with the six loop sections of the page:

```text
scroll ↓            the arm…                                the page section on screen
─────────────────────────────────────────────────────────────────────────────────────
hero                idles; part waits at the head of line   A working robot, from a sentence
01 DESIGN           picks the part; it solidifies           The agent writes a genome
02 SIMULATE         carries it into the test cell; drop     Every build is an experiment
03 TRAIN            re-grips it — twice wobbly, once clean  We train the will too
04 VERIFY           passes it under the scanner arch        If it isn't proven, it doesn't export
05 EXPORT           places it into the crate                It ends as a shopping list
06 MEMORY           a ghost copy returns to line start      Robot №100 is born smarter
CTA                 arm returns to rest, faces viewer       Your first robot is one sentence away
```

Scroll **is** the timeline: scrolling forward advances the arm's motion, scrolling backward
rewinds it frame-perfectly. The viewer never loses scroll control — the arm follows the page,
never the reverse.

Why this sells Virturoid specifically: the arm is *doing our actual pitch* — one artifact being
designed, simulated, trained, verified, exported, remembered. The animation is the architecture
diagram.

---

## 1. Decision record

| # | Decision | Choice | Why (alternatives rejected) |
| --- | --- | --- | --- |
| D1 | Rendering approach | **Real-time three.js scrubbing a Blender-baked glTF clip** | Image sequences (current SimFeed rig) can't sync a fixed frame count to six variable-height DOM sections, and payload grows with length (Apple built the custom "Flow" codec to cope). Procedural IK in code puts animation quality in the hardest place to art-direct. Spline/Rive: heavy runtime / 2D-only. |
| D2 | Animation authoring | **All motion baked in Blender, one clip** | Scrub = `mixer.setTime(t)`; zero runtime animation logic; art direction lives where the tools are. |
| D3 | Line orientation | **Belt runs diagonally into depth (lower-left near → upper-right far)** | Pure left→right reads as a 2D banner; pure depth hides travel. The diagonal shows both travel and the arm's reach, and the part's forward motion visually rhymes with downward scroll. |
| D4 | The part's identity | **One hero workpiece** (a ~14 cm anodized block with the Virturoid mark) whose *state* changes per station | One traveling protagonist gives narrative continuity; six separate blocks (earlier idea) dilute it. State changes carry the meaning (§3). |
| D5 | Pick/place mechanics in glTF | **Bake the part's world-space transform for the whole timeline as a root-level node** | glTF cannot animate re-parenting. Runtime attach/detach at markers breaks under backward scrub. With a fully-baked part track, "held vs free" is just data. |
| D6 | Pinning | **Fixed-position canvas + the page's existing `--run` scroll driver** | GSAP `pin: true` clones/wraps DOM and fights other layout (documented pain). We already compute global progress for the workstream/counters — one driver, no drift between motion systems. |
| D7 | Scrub smoothing | **dt-aware exponential catch-up in the render loop** (`t += (target−t)·(1−e^(−kΔt))`, k≈6) | Numeric-lag scrub without adding Lenis. Lenis (on one shared ticker) is the documented upgrade path if trackpad feel demands it — decision deferred to M3. |
| D8 | Compression | **meshopt (EXT_meshopt_compression) via glTF-Transform** | Decoder ≈ 30 KB vs Draco's ~300 KB wasm; faster decode; same tool also does `resample`/`quantize` for the dense baked animation. Blender's built-in Draco export is measurably worse than glTF-Transform post-processing. |
| D9 | Honesty | **The arm is labeled brand animation; the real MuJoCo SIM FEED moves into station 02 as "real capture"** | "No mockups" survives: we never present the stylized arm as simulator output. |

---

## 2. Scroll mechanics — the exact model

### 2.1 Two-level time mapping

Global scroll percent is too crude: copy edits would silently re-time every beat. Instead:

1. **Chapter table** (single source of truth, mirrors the clip's authored chapters):

```js
const CHAPTERS = [
  { id: "hero",     clip: [0.000, 0.060] },  // idle loop-ish hold
  { id: "design",   clip: [0.060, 0.210] },
  { id: "simulate", clip: [0.210, 0.360] },
  { id: "train",    clip: [0.360, 0.540] },  // longest beat — 3 grasp attempts
  { id: "verify",   clip: [0.540, 0.680] },
  { id: "export",   clip: [0.680, 0.830] },
  { id: "memory",   clip: [0.830, 0.960] },
  { id: "access",   clip: [0.960, 1.000] },  // rest pose, face viewer
];
```

2. **DOM measurement**: on load + `ResizeObserver` + `orientationchange`, measure each section's
   `[offsetTop, offsetTop + height]`. A section's **local progress** = how far its span has
   passed the viewport's 40% line (not the top — beats should play while the copy is readable,
   not after it leaves).

3. **Mapping**: `targetTime = lerp(chapter.clip[0], chapter.clip[1], smoothstep(localProgress)) × clipDuration`.
   `smoothstep` gives ease-in/out *within* a chapter; **hold plateaus** are authored into the clip
   itself (§6.4) so pausing mid-scroll always rests on a deliberate pose, and chapter boundaries
   tolerate imprecise scroll positions.

### 2.2 Catch-up loop & render-on-demand

```text
scroll event  → compute targetTime (cheap, no layout thrash: cached measurements)
rAF loop      → current += (target − current) · (1 − e^(−6·dt))
              → if |target − current| > 1e-4: mixer.setTime(current); render()
              → else: skip render entirely (idle GPU = 0)
visibility    → document.hidden ⇒ suspend rAF
```

- `action.play(); action.paused = true; mixer.timeScale = 1` once at load — then only `setTime`.
  (With `timeScale ≠ 1`, `setTime` remaps and desyncs — keep it at 1, always.)
- Backward scroll needs no special casing — the mixer is stateless under `setTime`, which is why
  D5 (fully baked part track) matters: no runtime attach state to un-wind.

### 2.3 Known failure modes, pre-answered

| Failure (documented in the wild) | Cause | Our mitigation |
| --- | --- | --- |
| "Animation off on mobile" | URL-bar show/hide resizes viewport, stale scroll ranges | re-measure on `resize`/`orientationchange`; `dvh` units; measurements cached, never read in the scroll handler |
| Twitchy scrub on wheel clicks | discrete wheel deltas 1:1 into time | D7 catch-up filter |
| Jank on fast fling | rendering every scroll event | scroll handler only writes `target`; rendering strictly rAF-gated |
| Page jumps around pinned section | ScrollTrigger pin DOM surgery | D6: fixed canvas, no pinning, no DOM cloning |
| Two motion systems drift | separate scroll math for reveals vs 3D | both read the same measurement cache & `--run` variable |

---

## 3. Choreography — station by station storyboard

Frame numbers assume the authored timeline of **1680 frames @ 24 fps** (70 s virtual). Every
chapter ends on a **hold pose** (≥ 12 frames flat) — the pose the viewer rests on while reading.

| Chapter | Frames | Arm action | Part state change | Set dressing active |
| --- | --- | --- | --- | --- |
| HERO | 0–100 | Rest pose, slow 2° sway; gripper open | Part waits at line head, **wireframe shell visible** | belt, idle status lamp |
| 01 DESIGN | 100–352 | Reach + grip part; lift to eye line; slow 90° wrist inspect | **Wireframe shell shrinks away → solid anodized body** (scale-keyed shells, §6.5) | — |
| 02 SIMULATE | 352–604 | Carry into test cell; **release mid-air**; part drops, bounces once, settles; arm re-grips | small scuff decal shell appears after bounce | test cell frame, floor grid flickers on |
| 03 TRAIN | 604–906 | **Three grasp attempts**: (1) part slips through closing gripper, (2) lifts then wobbles + set down, (3) clean confident lift-carry | — (motion IS the state) | small reward-lamp: red, red, **green** |
| 04 VERIFY | 906–1142 | Steady carry through scanner arch, deliberate pace | **stamp glyph shell appears** on part top face as it exits arch | arch emitter strip sweeps; lamp flips green |
| 05 EXPORT | 1142–1394 | Place into crate, tap-settle, withdraw | part seated in foam cutout | crate lid begins closing |
| 06 MEMORY | 1394–1613 | Arm points/gestures back up-line | **ghost duplicate** (translucent shell) slides from crate back to line head | line-head lamp lights |
| CTA | 1613–1680 | Return to rest, base rotates to face viewer, gripper open — an invitation | — | all lamps steady |

Choreography rules:

- **Reversibility**: every beat must read correctly scrubbed backwards (the SIMULATE bounce is a
  clean parabola; no particles, no motion blur, no secondary sim).
- **Readability at 300 px**: silhouettes over detail — key poses must read in the mobile-width
  stage. Test each hold pose as a thumbnail.
- **The arm has manners**: ease-in/out on every joint, anticipation before grips (2–3 frame
  pre-close), settle after places. Reference: industrial cobot demo reels, not character animation.

---

## 4. Visual & art direction

- **Palette**: matte gray links (`roughness 0.55–0.7`), near-black joints, **ember `#E8853B`**
  accent rings at each joint axis + gripper pads + part logo face. Set pieces in `#1A1B1F`.
  Background transparent — the page's paper `#101013` and workstream show through.
- **Lighting (runtime, not baked)**: one directional key (35° high, slightly warm), low cool
  ambient, ember rim from screen-left. `ACESFilmicToneMapping`, `outputColorSpace = SRGBColorSpace`,
  exposure ~1.1.
- **Shadows**: single 1024² shadow map from the key light, tight frustum on the work area — OR
  (cheaper, decide at M2) a baked AO card under belt/arm and no runtime shadows at all.
- **Camera**: 45 mm-equivalent, four keyed positions (hero / mid-line / scanner / crate) lerped
  by the same chapter map with heavy smoothing. Drift, don't fly: ≤ 15° total azimuth change.
  The arm is the actor; the camera is a tripod with feelings.
- **Scale on page**: desktop ≥ 1200 px — stage owns the right ~42%, text column left. The fixed
  canvas is full-viewport but the scene is framed right-of-center (camera offset), so text never
  overlaps geometry. 900–1200 px: scene recenters, sections overlay with backdrop panels.

---

## 5. Code architecture

```text
site/
  public/models/
    line_arm.glb            ← final compressed asset (user-built, §6)
    line_arm_poster.webp    ← hero-pose still, the LCP element
  src/components/
    RobotLine.astro         ← stage markup + poster + <script> module below
  src/lib/
    line-scrub.ts           ← measurement cache, chapter mapping, catch-up loop
    line-stage.ts           ← three.js scene: loader, lights, camera keys, render-on-demand
```

- **three.js payload discipline**: direct imports only (`three/src/…` tree-shakes poorly — use
  the top-level `three` module + `GLTFLoader` + `MeshoptDecoder`; measured budget §7). No react,
  no drei, no GSAP — the scrub math in §2 is ~40 lines.
- **Astro integration**: `RobotLine.astro` renders the fixed stage + poster immediately
  (server-rendered HTML, zero JS for first paint), then a module script lazily boots WebGL:
  `requestIdleCallback` → fetch glb (`fetchpriority=low`) → fade canvas over the poster.
  The page is fully readable before a single 3D byte arrives.
- **Debug surface**: `window.__lineDebug = { time, target, chapter, ranges }` — the test
  harness (§9) asserts against it.
- **Disposal**: single-page site — no teardown path needed beyond `visibilitychange` suspend.

---

## 6. Blender build sheet (the user-built deliverable)

### 6.1 Scene inventory

| Object (exact name) | Description | Tri budget |
| --- | --- | --- |
| `arm_root` | 6-DOF stylized arm: base turntable, 2 main links, forearm, wrist, 2-finger gripper | 28 000 |
| `part` | the hero workpiece: ~14 cm chamfered block, Virturoid mark on top face | 2 500 |
| `part_shell_wire` | wireframe overlay shell (design state) | 1 500 |
| `part_shell_scuff` | scuff decal shell (simulate state) | 300 |
| `part_shell_stamp` | verify stamp glyph shell | 300 |
| `part_ghost` | translucent duplicate (memory beat) | 2 500 |
| `belt` | line rail + rollers, runs lower-left → upper-right in depth | 9 000 |
| `test_cell` | simulate station: open frame + floor grid plane | 3 000 |
| `scanner_arch` | verify station gate with emitter strip | 3 500 |
| `crate` | export station: open crate, foam cutout, hinged lid | 4 000 |
| `lamp_reward`, `lamp_head` | small status lamps (emissive material swaps keyed by scale shells) | 400 |
| **Total** | | **≤ 55 000** |

### 6.2 Rig & bake

1. Author with IK (arm) + child-of constraints (part follows gripper while held) — freely.
2. **Bake everything to FK/world keys before export**: select armature + `part` + shells + lid →
   Object > Animation > **Bake Action** (visual keying, clear constraints, clear parents on
   `part` — it becomes a root-level node with a fully baked world-space track, per D5).
3. One action named **`line_loop`**, pushed to a single NLA strip. Delete all other actions.
4. Timeline **0–1680 @ 24 fps** with the chapter frame ranges of §3, including the ≥ 12-frame
  hold plateaus at every chapter boundary.

### 6.3 State changes without material animation

glTF can't animate material swaps — all part-state changes are **scale-keyed shells**:
a shell lives at scale 1 when visible, scale 0.0001 when hidden, keyed with constant
(stepped) interpolation. This survives every exporter and scrubs perfectly in reverse.

### 6.4 Materials & textures

- Principled BSDF only. Metal/rough workflow, **no procedural textures** (bake them).
- One shared **1024² atlas** (color + ORM packed) for everything except emissives; part logo
  and stamp glyph live in the atlas. Emissive strips/lamps: plain emissive colors, no texture.
- Ember `#E8853B`, paper-dark set pieces `#1A1B1F`, links neutral gray `#8A8D93`.

### 6.5 Export & post

1. File > Export > glTF 2.0 (`.glb`): +Y up · Apply Modifiers · Animation: **sampled**,
   only NLA strip `line_loop` · no cameras, no lights, no Draco (post handles compression).
2. Post-process (agent runs this, committed to repo):
   `npx @gltf-transform/cli optimize line_arm_raw.glb line_arm.glb --compress meshopt --texture-compress webp`
   (includes `resample` — collapses the dense bake — `prune`, `quantize`.)
3. QA gates before handoff: opens clean in https://gltf.report (0 errors), animation scrubs in
   the three.js editor, file **≤ 1.5 MB**.

---

## 7. Performance budgets

| Item | Budget | Enforcement |
| --- | --- | --- |
| `line_arm.glb` | ≤ 1.5 MB | §6.5 QA gate |
| three.js + loaders + decoder (gzipped) | ≤ 180 KB | `npm run build` size report |
| poster webp | ≤ 60 KB | capture script |
| LCP | < 2.5 s (poster is LCP; glb never blocks) | Lighthouse in §9 |
| CLS | ≈ 0 (canvas absolutely positioned, never reflows) | Lighthouse |
| scrub frame | ≤ 4 ms JS + render @ 1440p/1×DPR | perf probe §9 |
| idle | 0 renders/s when settled | `__lineDebug` assertion |

DPR capped at 1.5; antialias on; shadows only if the M2 lighting pass keeps frame budget.

---

## 8. Fallback ladder & accessibility

1. **`prefers-reduced-motion`**: poster only — no WebGL boot at all. (Also honored by the
   existing reveal/counter systems.)
2. **No WebGL / low memory** (`!WebGLRenderingContext` or `deviceMemory < 2`): poster only.
3. **Mobile < 900 px**: decide at M3 between (a) low-DPR canvas with the scene recentered
   behind panels, or (b) per-station posters crossfaded by the existing reveal system. Ship (b)
   first — it's free — and upgrade to (a) only if device testing is clean.
4. **Screen readers**: stage is `aria-hidden`; the six sections already carry the narrative.
5. **Keyboard scrolling** (PgDn/space jumps): the catch-up filter turns teleports into a fast
   but continuous playthrough — no special casing.

---

## 9. Testing & verification (extends `scripts/check_page.mjs`)

- **Sync assertion**: scroll to each section's midpoint → `__lineDebug.chapter === section.id`
  and `time` within the chapter's clip range.
- **Reversibility**: scrub to 80%, back to 20%, re-assert — no drift (`|time_expected − time| < ε`).
- **Idle assertion**: after 1 s without scroll, render counter stops incrementing.
- **Perf probe**: rAF-delta sampling during a scripted 8 s scroll — p95 frame < 16.7 ms.
- **Visual regression**: screenshot at each chapter hold pose → `vs_site/line_<chapter>.png`,
  eyeballed per iteration (same workflow as today's peek/check).
- **Fallback paths**: run once with `--force-prefers-reduced-motion`, once with WebGL disabled
  (`--disable-webgl`) — assert poster visible, zero console errors.
- Existing checks (no 404s, anchors, mobile overflow, docked feed removal) stay green.

---

## 10. Milestones

| # | Deliverable | Owner | Acceptance |
| --- | --- | --- | --- |
| **M0** | Scrub spike: any rigged `.glb` (even a cube arm) scrubbing via `--run` in an Astro island | agent | 60 fps desktop, reverse-clean, idle = 0 renders |
| **M1** | Chapter mapping wired to the six real sections; `__lineDebug`; test harness §9 | agent | sync assertions green through copy edits + resizes |
| **M2** | `line_arm.glb` per §6 integrated; lighting/camera pass; poster capture | **user (asset)** + agent | ≤ 1.5 MB; §3 beats read at 300 px; visual sign-off |
| **M3** | Fallback ladder, mobile decision, Lighthouse pass | agent | §7 budgets met on throttled mobile emulation |
| **M4** | Page swap: RobotLine in, SimFeed relocated into station 02 as "real capture" (D9) | agent | full check suite green; OG card re-captured |

Parallelism: M0/M1 need no asset — the user models against §6 while the harness is built, then
M2 is a drop-in.

---

## 11. Risks & open questions

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Asset quality misses (the whole point is "looks good") | high | §6 is prescriptive; M2 iterates against a live harness; hold poses reviewed as thumbnails early |
| Baked animation bloats the glb | med | `resample` + quantization; worst case reduce bake to 12 fps keys (scrub interpolates) |
| Text-over-3D legibility on mid widths | med | backdrop panels behind copy at < 1200 px (already the pattern for plates) |
| Scroll feel divides opinion (trackpad vs wheel vs touch) | med | k-factor tunable; Lenis as documented upgrade path (D7) |
| Two hero visuals compete (arm vs terminal) | low | terminal moves under the fold or into station 01 at M4 — decide at sign-off |

Open questions for the user before M2:
1. Arm style: match the reference build's proportions (tabletop 3-link) or a beefier 6-DOF
   industrial silhouette? (Plan assumes the latter.)
2. Should the part carry the Virturoid cube-mark or a neutral fiducial (ArUco-style)?
3. Belt props: minimal (rail only) or full set dressing (test cell, arch, crate) as specced?

---

## 12. Reference notes (research trail)

- three.js forum: scrub = `action.play(); action.paused = true;` + `mixer.setTime(clip.duration × progress)`
  on ScrollTrigger `onUpdate`; multiple mobile-desync reports traced to viewport resize.
- Production playbooks: prefer fixed canvas + tall scroll driver over ScrollTrigger `pin: true`
  (DOM cloning side-effects); numeric scrub lag > `scrub: true`; Lenis on GSAP's single ticker
  when smooth-scroll is wanted.
- Apple product pages: pre-rendered sequences require their custom "Flow" diff codec to stay
  light — evidence for D1's rejection of long image sequences.
- glTF-Transform discussions: Blender's built-in Draco underperforms; CLI `optimize` with
  meshopt/draco + `resample` is the standard pipeline (60 MB → 1.8 MB class results).
- glTF format limits (informed D5/§6.3): no animated re-parenting, no animated material swaps,
  no constraints — bake everything.
