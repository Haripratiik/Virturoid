// Screenshot individual sections at full size for design review.
//   node scripts/peek.mjs "#memory" out.png
import puppeteer from "puppeteer-core";
import { tmpdir } from "node:os";
import { join } from "node:path";

const sel = process.argv[2] ?? "#memory";
const name = process.argv[3] ?? "peek.png";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
await page.goto("http://localhost:4321/", { waitUntil: "networkidle2" });
const el = await page.$(sel);
await el.evaluate((n) => n.scrollIntoView());
await new Promise((r) => setTimeout(r, 2600)); // reveals + counters settle
const out = join(tmpdir(), "vs_site", name);
await el.screenshot({ path: out });
console.log(out);
await browser.close();
