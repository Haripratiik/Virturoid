# Virturoid landing page

Static Astro marketing site. It is deliberately separate from the desktop app and Python backend.

The page presents Virturoid as a self-improving robot-design brain: a stylized system view carries a generated robot through candidate, testing, verified, and banked-memory states. The stage is an illustration, never labeled as a live simulator capture. Real screenshots and metrics are clearly marked as artifact-backed evidence.

## Commands

```bash
npm install
npm run dev
npm run build
npm run preview
node scripts/check_page.mjs
```

`check_page.mjs` expects the preview server to be running. It verifies the four story states, reverse scrolling, the `/evidence` route, mobile state illustrations, console errors, and horizontal overflow. It writes screenshots to the system temp directory.

## Evidence and honesty

`/evidence` is the technical companion page. Its numbers render from `src/data/*.json`, which are unedited copies of real build artifacts. Real Studio images are in `public/shots/`; do not present the stylized system view as a recording or a physics result.

## Waitlist

The primary request-access action currently falls back to `mailto:hello@virturoid.dev`. Connect `PUBLIC_FORM_ENDPOINT` when a hosted form is ready.

## Deploy

`.github/workflows/deploy-site.yml` builds `site/` and deploys it to GitHub Pages on pushes to `main`.
