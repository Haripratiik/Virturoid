// Command-driven studio demo. Loads only when the page URL has ?demo=1.
//
// It does NOT auto-run. It idles on an empty studio and performs one "beat" at a time
// when told to -- either by the teleprompter remote (same-origin BroadcastChannel) or by
// the Left/Right arrow keys on this page. Each beat drives the SAME controls a presenter
// would (build via the assistant, switch packages / viewport modes / workspaces, focus
// the Reports/Artifacts tabs, select a memory node). No overlay is drawn, so a screen
// recording stays clean -- the Play/Pause/Next buttons live on the teleprompter.
//
// Beats 0..8 line up with teleprompter cards 1..9 (Open .. Limits). Cards 0 (title) and
// 10 (closing slides) have no studio action.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function runAutoDemo(api) {
  const { setWorkspace, setPackage, goTo, viewerSection, assistantPanel,
          selectMemoryNode, loadMemoryBuilds } = api;

  // A beat token cancels a still-running beat the moment a newer one starts, so fast
  // Next presses don't leave two beats fighting over the viewport.
  let beatToken = 0;
  const superseded = (tok) => tok !== beatToken;

  // ---- control helpers (reuse the app's own elements/handlers) ----
  async function ensureAssistant() {
    for (let i = 0; i < 50; i++) {
      if (assistantPanel._els && assistantPanel._els.input && assistantPanel._ctx) return true;
      await sleep(100);
    }
    return false;
  }
  async function typeInto(text, perChar = 42) {
    const input = assistantPanel._els.input;
    input.focus();
    input.value = "";
    for (const ch of text) { input.value += ch; input.dispatchEvent(new Event("input")); await sleep(perChar); }
  }
  function setMode(mode) {
    const r = viewerSection._refs;
    if (!r || !r.modeSelect) return;
    r.modeSelect.value = mode;
    r.modeSelect.dispatchEvent(new Event("change"));
  }
  async function pick(id, mode, tok) {
    setPackage(id);
    await sleep(400);
    if (tok != null && superseded(tok)) return;
    if (mode) { setMode(mode); await sleep(250); }
  }
  // The humanoid has no static model and its episode falls -- show it STANDING by
  // seeking the replay to frame 0 and pausing (drives the viewport scrubber).
  async function freezeAtStart(tok) {
    const r = viewerSection._refs;
    if (!r || !r.scrub) return;
    for (let i = 0; i < 30 && (!r.playbar || r.playbar.style.display === "none"); i++) {
      if (tok != null && superseded(tok)) return;
      await sleep(100);
    }
    if (tok != null && superseded(tok)) return;
    r.scrub.value = "0";
    r.scrub.dispatchEvent(new Event("input")); // pauses + applies frame 0
  }
  // cancellable hold used for the sub-paced beats (spec, any-body)
  function hold(tok, sec) {
    return new Promise((res) => {
      let t = 0;
      const iv = setInterval(() => {
        if ((tok != null && superseded(tok)) || t >= sec) { clearInterval(iv); res(); }
        t += 0.1;
      }, 100);
    });
  }
  const PROMPT = "build a quadruped robot that walks";

  // ---- the beats (0..8 == teleprompter cards 1..9) ----
  const BEATS = [
    { name: "open", run: async () => { setWorkspace("build"); await sleep(200); setPackage(null); } },
    { name: "build + walk", run: async (tok) => {
        setWorkspace("build"); await sleep(400);
        if (!(await ensureAssistant())) { console.warn("[autodemo] assistant not ready"); return; }
        if (superseded(tok)) return;
        await typeInto(PROMPT);
        if (superseded(tok)) return;
        await sleep(300);
        await assistantPanel._send(PROMPT);     // real live build (~10s); sets the package + opens the viewport
        if (superseded(tok)) return;
        setMode("episode");                     // force the walking replay (never rely on default mode)
      } },
    { name: "arm scene",   run: async (tok) => { await pick("arm_sort", "scene", tok); } },
    { name: "arm task",    run: async () => { setMode("episode"); } },
    { name: "spec",        run: async (tok) => { goTo("reports"); await hold(tok, 6); if (superseded(tok)) return; goTo("artifacts"); } },
    { name: "humanoid",    run: async (tok) => { await pick("humanoid", "episode", tok); await freezeAtStart(tok); } },
    { name: "mobile room", run: async (tok) => { await pick("build_a_mobile_base_that_delivers_parts_indoors", "scene", tok); } },
    { name: "hexapod",     run: async (tok) => { await pick("hexapod_walk", "episode", tok); } },
    { name: "memory",      run: async () => {
        setWorkspace("memory"); await sleep(800);
        try { await loadMemoryBuilds(); } catch (e) {}
        await sleep(300);
        try { selectMemoryNode("build:arm_sort"); } catch (e) {}
      } },
    { name: "limits",      run: async (tok) => {
        setWorkspace("build"); await sleep(400);
        await pick("humanoid", "episode", tok); await freezeAtStart(tok);
      } },
  ];

  let current = -1;
  async function gotoBeat(i) {
    if (i < 0 || i >= BEATS.length) return;
    const tok = ++beatToken;     // supersede any in-flight beat
    current = i;
    console.log(`[autodemo] beat ${i}  ${BEATS[i].name}`);
    try { await BEATS[i].run(tok); } catch (err) { console.warn(`[autodemo] beat ${i} (${BEATS[i].name}) error`, err); }
  }
  function resetStudio() { beatToken += 1; current = -1; setWorkspace("build"); setPackage(null); }

  // ---- remote: the teleprompter conducts over a same-origin BroadcastChannel ----
  const chan = ("BroadcastChannel" in window) ? new BroadcastChannel("virturoid-demo") : null;
  if (chan) {
    chan.onmessage = (e) => {
      const m = e.data || {};
      if (m.cmd === "goto" && typeof m.index === "number") gotoBeat(m.index);
      else if (m.cmd === "reset") resetStudio();
      else if (m.cmd === "hello") chan.postMessage({ cmd: "demo-ready", beats: BEATS.length });
    };
    chan.postMessage({ cmd: "demo-ready", beats: BEATS.length });
  }

  // ---- keyboard fallback on the studio page (works without the teleprompter) ----
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { gotoBeat(Math.min(BEATS.length - 1, current + 1)); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { gotoBeat(Math.max(0, current - 1)); e.preventDefault(); }
    else if (e.key === "Home") { resetStudio(); }
  });

  resetStudio();
  console.log(`[autodemo] ready (${BEATS.length} beats) -- waiting for the teleprompter remote, or use Left/Right arrows here`);
}
