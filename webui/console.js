import { el, clear } from "./util.js";

function timestamp(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour12: false });
}

export const consolePanel = {
  id: "console",
  title: "Console",
  dependsOnPackage: false,
  _log: null,
  _bound: false,

  mount(elx, ctx) {
    clear(elx);
    const toolbar = el("div", { class: "console-bar" }, [
      el("span", { class: "console-title", text: "EVENT LOG" }),
      el("button", { class: "console-clear", type: "button", text: "Clear", onClick: () => clear(this._log) }),
    ]);
    this._log = el("div", { class: "console-log" });
    elx.appendChild(el("div", { class: "console" }, [toolbar, this._log]));

    this._write("Virturoid Studio ready.", "ok");

    if (!this._bound) {
      ctx.bus.addEventListener("log", (e) => this._write(e.detail.message, e.detail.kind, e.detail.ts));
      ctx.bus.addEventListener("assistant-status", (e) => {
        const s = e.detail;
        if (s && s.online && s.model_available) this._write(`Local model online: ${s.model}`, "ok");
      });
      this._bound = true;
    }
  },

  _write(message, kind = "info", ts = Date.now()) {
    if (!this._log) return;
    this._log.appendChild(el("div", { class: `console-line ${kind}` }, [
      el("span", { class: "console-ts", text: timestamp(ts) }),
      el("span", { class: "console-msg", text: message }),
    ]));
    this._log.scrollTop = this._log.scrollHeight;
  },
};
