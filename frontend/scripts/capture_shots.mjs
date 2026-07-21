// Captures REAL Studio screenshots for the landing page (site/public/shots).
// Tries hardware GL first so the robot actually renders in the viewport.
//   node scripts/capture_shots.mjs [baseUrl]
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const BASE = process.argv[2] ?? "http://127.0.0.1:8765";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = resolve(import.meta.dirname, "..", "..", "site", "public", "shots");
mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  // No --disable-gpu: let Chrome use ANGLE/D3D so WebGL renders the robot.
  args: ["--no-sandbox", "--hide-scrollbars", "--window-size=1760,1100", "--use-angle=default"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1760, height: 1100, deviceScaleFactor: 1.25 });

async function open(params) {
  await page.goto(`${BASE}/studio?robot=arm_sort&tour=0&${params}`, { waitUntil: "networkidle2" });
  await sleep(2200); // robot mesh load + first frames
  await page.keyboard.press("f"); // frame the robot
  await sleep(700);
}

async function zoomIn(amount) {
  const canvas = await page.$("canvas");
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel({ deltaY: amount });
  await sleep(900);
}

// 1) Hero — full studio, Design workspace, Scene mode (the robot in its
// physics test scene reads far better than the bare-URDF rest pose).
await open("ws=design");
const heroScene = await page.$('xpath/.//button[@aria-pressed][contains(., "Scene")]');
if (heroScene) {
  await heroScene.click();
  await sleep(2500);
  await page.keyboard.press("f");
  await sleep(700);
}
await zoomIn(-500);
const gl = await page.evaluate(() => {
  const c = document.createElement("canvas");
  const g = c.getContext("webgl2") ?? c.getContext("webgl");
  if (!g) return "none";
  const info = g.getExtension("WEBGL_debug_renderer_info");
  return info ? g.getParameter(info.UNMASKED_RENDERER_WEBGL) : "unknown";
});
console.log(`WebGL renderer: ${gl}`);
await page.screenshot({ path: `${OUT}/studio-hero.png` });
console.log("studio-hero.png");

// 2) Agent panel element shot.
const agent = await page.$('section[aria-label="Agent"]');
if (agent) {
  await agent.screenshot({ path: `${OUT}/studio-agent.png` });
  console.log("studio-agent.png");
}

// 3) Inspector with Anatomy open.
const anatomyTab = await page.$('xpath/.//button[@role="tab"][contains(., "Anatomy")]');
if (anatomyTab) {
  await anatomyTab.click();
  await sleep(500);
}
const inspector = await page.$('section[aria-label="Inspector"]');
if (inspector) {
  await inspector.screenshot({ path: `${OUT}/studio-inspector.png` });
  console.log("studio-inspector.png");
}

// 4) Simulate workspace — switch to Scene mode so the physics test scene
// (table, blocks, bins) is what the shot shows.
await open("ws=simulate");
const sceneChip = await page.$('xpath/.//button[@aria-pressed][contains(., "Scene")]');
if (sceneChip) {
  await sceneChip.click();
  await sleep(2500);
  await page.keyboard.press("f");
  await sleep(700);
  await zoomIn(-400);
}
await page.screenshot({ path: `${OUT}/studio-simulate.png` });
console.log("studio-simulate.png");

await browser.close();
console.log(`done -> ${OUT}`);
