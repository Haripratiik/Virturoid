# Replacing the SIM FEED frames with Blender renders

The hero's scroll-scrubbed robot feed reads frames from `public/seq/frame_000.webp … frame_043.webp`.
Today they're captured from the real simulator (`frontend/scripts/capture_seq.mjs`). To upgrade the
visuals, render a sequence in Blender and drop the files in with the **same names** — zero code changes.
(If you change the frame count, update `FRAMES` in `src/components/SimFeed.astro`.)

## Render spec

- **Frames**: 44 (more is fine — 60–90 for extra smoothness; keep total < 6 MB)
- **Resolution**: 1500 × 1000 (3:2), rendered at 100%, **WebP quality ~70** (or PNG → convert)
- **Background**: solid dark `#101013` (matches the feed panel; do NOT use transparency)
- **Naming**: `frame_000.webp`, zero-padded 3 digits, sequential

## Scene direction (so it matches the page's story)

The page walks the loop: design → simulate → train → verify → export. Choreograph the sequence so
scrolling *runs a task attempt*:

1. **Frames 0–10**: camera orbits in from a 3/4 high angle; arm at rest at its table. Red + blue
   cubes (~30 mm) scattered mid-table, two open bins (one red-rimmed, one blue-rimmed).
2. **Frames 10–24**: arm reaches, closes gripper on the red cube (camera keeps orbiting slowly ~90° total).
3. **Frames 24–36**: carry + place into the red bin; brief settle.
4. **Frames 36–43**: arm retracts to rest; camera pushes in on the completed sort.

## Style guide (keep it "instrument", not "product ad")

- Matte gray links (roughness 0.6+), **no glossy studio floor**, no bloom
- Ember-orange `#E8853B` accent: thin ring or marker at each joint axis
- Faint floor grid (0.5 m squares), barely visible
- One soft key light + dim cool fill; shadows soft but present
- Optional: tiny red **contact marker crosses** where the gripper touches the cube (on ~3 frames)

## Camera

- Focal ~50 mm, target the table center, radius ≈ 1.9 m → 1.25 m over the sequence
- Elevation ~24° → ~16°; total azimuth sweep ≈ 90–95°
- No motion blur (frames are scrubbed both directions)

Export check: `npm run build && npm run preview`, scroll — the feed should scrub your render.
