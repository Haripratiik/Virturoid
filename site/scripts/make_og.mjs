// Captures the social share card (og.png, 1200x630) from the live hero.
//   npm run preview  →  node scripts/make_og.mjs
import puppeteer from "puppeteer-core";
import { resolve } from "node:path";

const URL = "http://localhost:4321";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = resolve(import.meta.dirname, "..", "public", "og.png");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 1 });
await page.goto(URL, { waitUntil: "networkidle2" });
await sleep(5200); // typing demo + sim feed first frame
await page.screenshot({ path: OUT });
await browser.close();
console.log(`og card -> ${OUT}`);
