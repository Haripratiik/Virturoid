// Landing page render check: desktop + mobile full-page screenshots, console
// error detection, anchor-link and reveal-animation sanity.
//   node scripts/check_page.mjs [url]
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL = process.argv[2] ?? "http://localhost:4321/";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = join(tmpdir(), "vs_site");
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errors = [];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--hide-scrollbars", "--use-angle=default"],
});
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("response", (r) => r.status() === 404 && errors.push(`404: ${r.url()}`));

// Desktop
await page.setViewport({ width: 1440, height: 900 });
await page.goto(URL, { waitUntil: "networkidle2" });
await sleep(4500); // let the hero typing demo finish
await page.screenshot({ path: join(OUT, "desktop_hero.png") });
// Force reveals for the full-page shot (IntersectionObserver needs scroll).
await page.evaluate(() => document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in")));
await sleep(400);
await page.screenshot({ path: join(OUT, "desktop_full.png"), fullPage: true });

// Anchor nav works?
await page.click('a[href="#verify"]');
await sleep(2000); // smooth scroll needs time on a long page
const anchorState = await page.evaluate(() => {
  const el = document.getElementById("verify");
  return { found: !!el, top: el ? Math.round(el.getBoundingClientRect().top) : null, scrollY: Math.round(window.scrollY), hash: location.hash };
});
console.log(`anchor state: ${JSON.stringify(anchorState)}`);
if (!anchorState.found || anchorState.top < -200 || anchorState.top > 300)
  errors.push(`anchor #gates did not scroll into view (${JSON.stringify(anchorState)})`);
// Mid-page: the sim feed should be docked to the corner and scrubbed forward.
await page.screenshot({ path: join(OUT, "desktop_mid.png") });
const docked = await page.evaluate(() => document.getElementById("simfeed")?.classList.contains("docked"));
if (!docked) errors.push("sim feed did not dock on scroll");

// Mobile
await page.setViewport({ width: 390, height: 844, isMobile: true, deviceScaleFactor: 2 });
await page.goto(URL, { waitUntil: "networkidle2" });
await sleep(4200);
await page.evaluate(() => document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in")));
await sleep(400);
await page.screenshot({ path: join(OUT, "mobile_full.png"), fullPage: true });

// Horizontal overflow check (mobile) — the classic responsive bug.
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
if (overflow > 2) errors.push(`mobile horizontal overflow: ${overflow}px`);

await browser.close();
console.log(`shots -> ${OUT}`);
if (errors.length) {
  console.log(`${errors.length} ISSUE(S):`);
  errors.forEach((e) => console.log(` - ${e}`));
  process.exitCode = 1;
} else {
  console.log("clean: no console errors, no 404s, anchors ok, no mobile overflow");
}
