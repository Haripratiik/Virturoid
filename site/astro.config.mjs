import { defineConfig } from "astro/config";

// Static marketing site. Deployed to GitHub Pages by .github/workflows/deploy-site.yml.
// `base` is injected by the workflow for project pages (user.github.io/<repo>);
// locally it stays "/" so `astro dev`/`preview` behave normally.
export default defineConfig({
  output: "static",
  site: process.env.SITE_URL ?? "http://localhost:4321",
  base: process.env.SITE_BASE ?? "/",
});
