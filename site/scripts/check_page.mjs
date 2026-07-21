// Landing page verification: real product proof, evidence route, and responsive layout.
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE_URL = process.argv[2] ?? "http://127.0.0.1:4321/";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = join(tmpdir(), "vs_site");
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const errors = [];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--hide-scrollbars", "--use-angle=default"],
});
const page = await browser.newPage();
page.on("console", (message) => message.type() === "error" && errors.push(`console: ${message.text()}`));
page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
page.on("response", (response) => response.status() === 404 && errors.push(`404: ${response.url()}`));

async function goTo(id) {
  await page.evaluate((target) => {
    const el = document.getElementById(target);
    if (!el) return;
    window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top + 96, behavior: "instant" });
  }, id);
  await sleep(350);
}

await page.setViewport({ width: 1440, height: 900 });
await page.goto(BASE_URL, { waitUntil: "networkidle2" });
await sleep(300);
await page.screenshot({ path: join(OUT, "desktop_hero.png") });

const hero = await page.evaluate(() => ({
  heroCapture: !!document.querySelector(".hero-capture img[src='shots/studio-hero.png']"),
  robotGallery: !!document.querySelector(".robot-bridge .robot-gallery img"),
  syntheticStage: !!document.getElementById("simulation-canvas") || !!document.getElementById("simulation-workcell"),
  forbiddenCopy: ["LIVE REPLAY", "SIM FEED", "NO MOCKUPS"].filter((text) => document.body.innerText.includes(text)),
}));
if (!hero.heroCapture) errors.push("hero is missing the real Studio capture");
if (!hero.robotGallery) errors.push("landing page is missing the generated robot-output proof surface");
if (hero.syntheticStage) errors.push("synthetic robot stage is still present");
if (hero.forbiddenCopy.length) errors.push(`retired or misleading copy is visible: ${hero.forbiddenCopy.join(", ")}`);

await goTo("robots");
await page.screenshot({ path: join(OUT, "section_robots.png") });

for (const [id, selector] of [["job", ".agent-capture img"], ["world", ".world-capture img"], ["proof", ".ledger"], ["memory", ".archive"]]) {
  await goTo(id);
  const present = await page.$(selector);
  if (!present) errors.push(`${id} is missing its expected proof surface: ${selector}`);
  await page.screenshot({ path: join(OUT, `section_${id}.png`) });
}

const proofLink = await page.$('a[href="/evidence"]');
if (!proofLink) errors.push("missing evidence route link");
await page.goto(new URL("evidence", BASE_URL).href, { waitUntil: "networkidle2" });
const evidenceTitle = await page.$eval("h1", (el) => el.textContent);
if (!evidenceTitle?.includes("prove today")) errors.push("evidence route did not render expected content");
await page.screenshot({ path: join(OUT, "evidence.png"), fullPage: true });

await page.setViewport({ width: 390, height: 844, isMobile: true, deviceScaleFactor: 2 });
await page.goto(BASE_URL, { waitUntil: "networkidle2" });
await sleep(250);
const mobile = await page.evaluate(() => ({
  overflow: document.documentElement.scrollWidth - window.innerWidth,
  heroCapture: !!document.querySelector(".hero-capture"),
  robotGallery: !!document.querySelector(".robot-bridge .robot-gallery"),
  sections: ["job", "world", "proof", "memory"].filter((id) => !!document.getElementById(id)).length,
}));
if (mobile.overflow > 2) errors.push(`mobile horizontal overflow: ${mobile.overflow}px`);
if (!mobile.heroCapture) errors.push("mobile is missing the real Studio hero capture");
if (!mobile.robotGallery) errors.push("mobile is missing the robot-output proof surface");
if (mobile.sections !== 4) errors.push(`expected four landing chapters, got ${mobile.sections}`);
await page.screenshot({ path: join(OUT, "mobile_hero.png") });

await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
await page.setViewport({ width: 1440, height: 900 });
await page.goto(BASE_URL, { waitUntil: "networkidle2" });
const reduced = await page.evaluate(() => ({
  syntheticStage: !!document.getElementById("simulation-canvas") || !!document.getElementById("simulation-workcell"),
  heroCapture: !!document.querySelector(".hero-capture"),
}));
if (reduced.syntheticStage) errors.push("a synthetic animated stage remains in reduced-motion mode");
if (!reduced.heroCapture) errors.push("reduced-motion page is missing the static product proof");
await page.screenshot({ path: join(OUT, "reduced_motion.png") });

await browser.close();
console.log(`shots -> ${OUT}`);
if (errors.length) {
  console.log(`${errors.length} ISSUE(S):`);
  errors.forEach((error) => console.log(` - ${error}`));
  process.exitCode = 1;
} else {
  console.log("clean: real captures, evidence route, responsive layout, and reduced-motion fallback verified");
}
