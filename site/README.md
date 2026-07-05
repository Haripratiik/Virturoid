# Virturoid landing page

Static marketing site (Astro). Separate from the app by design — nothing here is imported by `frontend/` or the Python backend.

## Commands

```bash
npm install
npm run dev       # local dev at :4321
npm run build     # static output in dist/
npm run preview   # serve the built site
node scripts/check_page.mjs   # automated render check (desktop + mobile, needs the preview running)
node scripts/peek.mjs "#train" out.png   # full-res capture of one section
node scripts/make_og.mjs      # regenerate the social card from the live hero
```

## Sim feed (scroll-scrubbed robot)

The hero's SIM FEED scrubs `public/seq/frame_*.webp` with scroll and docks to the corner
while you read. Frames are captured from the real simulator by
`frontend/scripts/capture_seq.mjs` (Studio server must be running). To replace them with
Blender renders, follow `ASSETS.md` — same filenames, zero code changes.

## Honesty rule

Every number on the page renders from `src/data/*.json`, which are **unedited copies of real build artifacts** (`build/ui_verify/arm_sort`). If you regenerate the reference build, re-copy the artifacts — never hand-edit the values. The screenshots in `public/shots/` are captured from the real Studio by `frontend/scripts/capture_shots.mjs`.

## Waitlist form

Set `PUBLIC_FORM_ENDPOINT` in `site/.env` to a Formspree endpoint (e.g. `https://formspree.io/f/xxxx`). Without it the page falls back to a `mailto:` link.

## Deploy

`.github/workflows/deploy-site.yml` builds `site/` and deploys to GitHub Pages on push to `main` (Settings → Pages → Source: GitHub Actions).
