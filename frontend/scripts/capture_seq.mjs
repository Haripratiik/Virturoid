// Renders the landing page's scroll-scrub sequence from the REAL simulator:
// orbits the camera around the actual MuJoCo scene and saves webp frames.
// Output: site/public/seq/frame_000.webp ... — replace later with Blender
// renders (same names) without touching site code. See site/ASSETS.md.
//   node scripts/capture_seq.mjs
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const BASE = "http://127.0.0.1:8765";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = resolve(import.meta.dirname, "..", "..", "site", "public", "seq");
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const FRAMES = 44;

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--hide-scrollbars", "--window-size=1500,1000", "--use-angle=default"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1000, deviceScaleFactor: 1 });
await page.goto(`${BASE}/studio?robot=arm_sort&ws=design&tour=0`, { waitUntil: "networkidle2" });
await sleep(2000);

// Maximum canvas: close both side panels and the console dock.
for (const sel of ['button[aria-label="Close Inspector"]', 'button[aria-label="Close Agent"]']) {
  const b = await page.$(sel);
  if (b) await b.click();
}
await page.keyboard.down("Control");
await page.keyboard.press("j");
await page.keyboard.up("Control");
await sleep(400);

// Scene mode (table + red/blue blocks), framed.
const sceneChip = await page.$('xpath/.//button[@aria-pressed][contains(., "Scene")]');
if (sceneChip) {
  await sceneChip.click();
  await sleep(2600);
}
await page.keyboard.press("f");
await sleep(800);

// No UI chrome in the frames: hide status chips/tooltips, park the mouse.
await page.addStyleTag({ content: '[role="tooltip"], [role="status"] { display: none !important; }' });
await page.mouse.move(8, 990);
await sleep(300);

// Freeze damping so direct camera placement sticks.
await page.evaluate(() => {
  const e = window.__virturoidEngine;
  e.controls.enableDamping = false;
});

const canvas = await page.$("canvas");
for (let i = 0; i < FRAMES; i++) {
  const t = i / (FRAMES - 1);
  await page.evaluate(
    ([t]) => {
      const e = window.__virturoidEngine;
      const c = e.controls.target;
      const az = 2.35 - t * 1.65; // orbit ~95°
      const el = 0.42 - t * 0.14; // settle toward tabletop height
      const r = 1.9 - t * 0.65; // slow push-in, tight on the workcell
      e.camera.position.set(
        c.x + r * Math.cos(el) * Math.cos(az),
        c.y + r * Math.sin(el),
        c.z + r * Math.cos(el) * Math.sin(az),
      );
      e.controls.update();
    },
    [t],
  );
  await sleep(90);
  await canvas.screenshot({
    path: `${OUT}/frame_${String(i).padStart(3, "0")}.webp`,
    type: "webp",
    quality: 68,
  });
  process.stdout.write(`\rframe ${i + 1}/${FRAMES}`);
}
await browser.close();
console.log(`\ndone -> ${OUT}`);
