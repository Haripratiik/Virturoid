// Interaction smoke test: drives the built Studio UI in headless Chrome,
// clicking through tabs, toggles, popovers and panel-docking controls, and
// saves a screenshot after each step. Fails loudly on page console errors.
//
//   node scripts/clickthrough.mjs [baseUrl]
//
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = process.argv[2] ?? "http://127.0.0.1:8765";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = join(tmpdir(), "vs_clickthrough");
mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const xp = (expr) => `xpath/.${expr}`;

const errors = [];
let step = 0;

async function shot(page, name) {
  step += 1;
  const file = join(OUT, `ct_${String(step).padStart(2, "0")}_${name}.png`);
  await page.screenshot({ path: file });
  console.log(`shot ${file}`);
}

async function click(page, selector, name) {
  const el = await page.$(selector);
  if (!el) {
    errors.push(`MISSING CONTROL: ${name} (${selector})`);
    console.log(`!! missing: ${name}`);
    return false;
  }
  await el.click();
  await sleep(450);
  return true;
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--window-size=1720,980"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1720, height: 980 });
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console.error: ${m.text()}`);
});
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("response", (r) => {
  if (r.status() === 404) console.log(`404: ${r.url()}`);
});

await page.goto(`${BASE}/studio?robot=arm_sort&ws=design&tour=0`, { waitUntil: "networkidle2" });
await sleep(1500);
await shot(page, "design");

// Camera navigation: orbit (left-drag), zoom (wheel), pan (right-drag) must
// actually move the camera. Verified through the engine test hook.
const camState = () =>
  page.evaluate(() => {
    const e = window.__virturoidEngine;
    if (!e) return null;
    return { pos: e.camera.position.toArray(), target: e.controls.target.toArray() };
  });
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
{
  const canvas = await page.$("canvas");
  const box = canvas ? await canvas.boundingBox() : null;
  const s0 = await camState();
  if (!box || !s0) {
    errors.push("NAV: no canvas or engine hook");
  } else {
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    // orbit
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 140, cy + 70, { steps: 10 });
    await page.mouse.up();
    await sleep(600);
    const s1 = await camState();
    if (dist(s0.pos, s1.pos) < 0.01) errors.push("NAV: orbit did not move the camera");
    // zoom
    await page.mouse.move(cx, cy);
    await page.mouse.wheel({ deltaY: -480 });
    await sleep(600);
    const s2 = await camState();
    if (Math.abs(dist(s1.pos, s1.target) - dist(s2.pos, s2.target)) < 0.01)
      errors.push("NAV: wheel zoom did not change camera distance");
    // pan
    await page.mouse.move(cx, cy);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(cx + 120, cy + 40, { steps: 10 });
    await page.mouse.up({ button: "right" });
    await sleep(600);
    const s3 = await camState();
    if (dist(s2.target, s3.target) < 0.005) errors.push("NAV: right-drag pan did not move the target");
    console.log(
      `nav: orbit ${dist(s0.pos, s1.pos).toFixed(3)} · zoom ${(dist(s1.pos, s1.target) - dist(s2.pos, s2.target)).toFixed(3)} · pan ${dist(s2.target, s3.target).toFixed(3)}`,
    );
  }
}

// Tooltip delay: hover an editor tab, wait past the 700ms rest delay.
const simTab = await page.$(xp(`//button[@role="tab"][contains(., "Simulate")]`));
if (simTab) {
  await simTab.hover();
  await sleep(1100);
  await shot(page, "tooltip_after_delay");
}

// Edge tooltips must clamp inside the viewport, not clip at screen edges:
// right end of the viewport toolbar (bottom side) + far corners of the
// status bar (top side).
for (const [sel, name] of [
  ['button[aria-label="Viewport help"]', "tooltip_edge_right"],
  ['button[aria-label="Toggle wireframe"]', "tooltip_edge_wire"],
]) {
  const el = await page.$(sel);
  if (el) {
    await el.hover();
    await sleep(1100);
    await shot(page, name);
    const clipped = await page.evaluate(() => {
      const t = document.querySelector('[role="tooltip"]');
      if (!t) return "no tooltip rendered";
      const r = t.getBoundingClientRect();
      return r.right > window.innerWidth || r.left < 0 || r.bottom > window.innerHeight || r.top < 0
        ? `CLIPPED: ${JSON.stringify(r)}`
        : "ok";
    });
    console.log(`${name}: ${clipped}`);
    if (clipped !== "ok") errors.push(`${name}: ${clipped}`);
    await page.mouse.move(600, 400);
    await sleep(200);
  } else {
    errors.push(`MISSING CONTROL: ${name} (${sel})`);
  }
}

// Inspector tabs.
for (const tab of ["Anatomy", "Parts", "Sensors", "Properties"]) {
  await click(page, xp(`//button[@role="tab"][contains(., "${tab}")]`), `inspector ${tab}`);
  await shot(page, `inspector_${tab.toLowerCase()}`);
}

// Viewport display toggles + help popover.
await click(page, 'button[aria-label="Toggle wireframe"]', "wireframe");
await click(page, 'button[aria-label="Toggle axes"]', "axes");
await shot(page, "viewport_toggles");
await click(page, 'button[aria-label="Viewport help"]', "viewport help");
await shot(page, "viewport_help");
await click(page, 'button[aria-label="Viewport help"]', "viewport help close");

// Camera presets (state change only; canvas is black in headless GL).
await click(page, xp(`//button[text()="front"]`), "camera front");

// Dock tabs.
for (const tab of ["Jobs", "Artifacts", "Data", "Console"]) {
  await click(page, xp(`//button[contains(., "${tab}")][ancestor::*[@aria-label="Console panel" or true()]]`), `dock ${tab}`);
  await shot(page, `dock_${tab.toLowerCase()}`);
}

// Move the Agent panel to the left (movable panels), then back.
await click(page, 'button[aria-label="Move Agent to the left side"]', "move agent left");
await shot(page, "agent_docked_left");
await click(page, 'button[aria-label="Move Agent to the right side"]', "move agent right");

// Editor tabs.
for (const tab of ["Simulate", "Train", "Verify", "Library", "Design"]) {
  await click(page, xp(`//button[@role="tab"][contains(., "${tab}")]`), `workspace ${tab}`);
  await shot(page, `ws_${tab.toLowerCase()}`);
}

// Command palette.
await page.keyboard.down("Control");
await page.keyboard.press("k");
await page.keyboard.up("Control");
await sleep(400);
await shot(page, "palette");
await page.keyboard.press("Escape");

// Welcome tour: open from the activity bar, page through it.
await click(page, xp(`//button[starts-with(@aria-label, "Welcome tour")]`), "tour open");
console.log("tour dialog present:", !!(await page.$('[role="dialog"]')));
await shot(page, "tour_step1");
await click(page, xp(`//button[contains(., "Next")]`), "tour next");
await shot(page, "tour_step2");
await click(page, 'button[aria-label="Skip the tour"]', "tour close");

// Agent panel: close via X, reopen from the activity bar.
await click(page, 'button[aria-label="Close Agent"]', "close agent");
await shot(page, "agent_closed");
await click(page, xp(`//button[starts-with(@aria-label, "Agent")]`), "reopen agent");
await shot(page, "agent_reopened");

await browser.close();

console.log(`\n${step} screenshots -> ${OUT}`);
if (errors.length) {
  console.log(`\n${errors.length} ISSUE(S):`);
  for (const e of errors) console.log(` - ${e}`);
  process.exitCode = 1;
} else {
  console.log("no console errors, all controls found");
}
