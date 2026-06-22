import { el, clear, escapeHtml } from "./util.js";

const SUGGESTIONS = [
  "Build a tabletop arm that sorts red and blue blocks",
  "Build a mobile base that delivers parts indoors",
  "How do I improve this robot's readiness?",
  "Explain domain randomization for grasping",
];

function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br/>");
}

export const assistantPanel = {
  id: "assistant",
  title: "Assistant",
  dependsOnPackage: false,
  _ctx: null,
  _history: [],
  _els: null,
  _busy: false,

  mount(elx, ctx) {
    this._ctx = ctx;
    clear(elx);

    const head = el("div", { class: "asst-head" }, [
      el("div", { class: "asst-head-title" }, [
        el("span", { class: "asst-badge", text: "VS" }),
        el("div", {}, [
          el("h3", { text: "Build Assistant" }),
          el("span", { class: "asst-sub", text: "describe it \u2192 it builds it" }),
        ]),
      ]),
      el("button", { class: "asst-newchat", type: "button", title: "New chat", onClick: () => this._reset() }, "New chat"),
    ]);

    const statusBar = el("div", { class: "asst-status" }, [
      el("span", { class: "asst-dot" }),
      el("span", { class: "asst-status-text", text: "checking local model..." }),
    ]);
    const log = el("div", { class: "asst-log" });
    const suggestions = el("div", { class: "asst-suggestions" });
    const input = el("textarea", { class: "asst-input", rows: "1", placeholder: "Ask, or describe a robot to build..." });
    const send = el("button", { class: "button primary asst-send", type: "button" }, "Send");
    const inputRow = el("div", { class: "asst-input-row" }, [input, send]);

    elx.appendChild(el("div", { class: "asst" }, [head, statusBar, log, suggestions, inputRow]));
    this._els = { log, input, send, suggestions, statusText: statusBar.querySelector(".asst-status-text"), statusBar };

    const submit = () => this._send(input.value);
    send.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
    });
    input.addEventListener("input", () => this._autosize());

    ctx.bus.addEventListener("assistant-status", (e) => this._renderStatus(e.detail));
    this._renderStatus(ctx.assistant.status);

    if (!this._history.length) {
      this._append("assistant",
        "I'm your build assistant. Describe a robot and I'll build it, then ask me to inspect readiness, scenes, or how to improve it.");
    } else {
      this._history.forEach((m) => this._renderMessage(m.role, m.content, m.build));
    }
    this._renderSuggestions();
  },

  onShow() { if (this._els) { this._els.input.focus(); this._autosize(); } },

  _reset() {
    this._history = [];
    if (!this._els) return;
    clear(this._els.log);
    this._append("assistant", "New session. What should we build?");
    this._renderSuggestions();
  },

  _autosize() {
    const ta = this._els && this._els.input;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(140, ta.scrollHeight) + "px";
  },

  _renderSuggestions() {
    if (!this._els) return;
    const box = clear(this._els.suggestions);
    const hasUser = this._history.some((m) => m.role === "user");
    if (hasUser) { box.style.display = "none"; return; }
    box.style.display = "flex";
    SUGGESTIONS.forEach((s) => {
      box.appendChild(el("button", { class: "asst-chip", type: "button", onClick: () => this._send(s) }, s));
    });
  },

  _renderStatus(status) {
    if (!this._els) return;
    const online = status && status.online && status.model_available;
    this._els.statusBar.className = `asst-status ${online ? "online" : "offline"}`;
    if (!status) this._els.statusText.textContent = "model status unknown";
    else if (online) this._els.statusText.textContent = `local model online \u00b7 ${status.model}`;
    else if (status.online) this._els.statusText.textContent = `Ollama up \u00b7 pull "${status.model}" for full chat`;
    else this._els.statusText.textContent = "local model offline \u00b7 deterministic build mode";
  },

  _append(role, content, build) {
    this._history.push({ role, content, build });
    this._renderMessage(role, content, build);
  },

  _renderMessage(role, content, build) {
    const msg = el("div", { class: `asst-msg ${role}` }, [
      el("div", { class: "asst-avatar", text: role === "user" ? "you" : "VS" }),
      el("div", { class: "asst-bubble", html: renderMarkdown(content) }),
    ]);
    if (build && build.package_url_prefix) {
      const name = String(build.package_url_prefix).split("/").filter(Boolean).pop();
      msg.querySelector(".asst-bubble").appendChild(el("button", {
        class: "button asst-open", type: "button",
        onClick: () => this._ctx.goTo("viewport"),
      }, `Open ${name} in viewport`));
    }
    this._els.log.appendChild(msg);
    this._els.log.scrollTop = this._els.log.scrollHeight;
  },

  async _send(text) {
    const value = (text || "").trim();
    if (!value || this._busy) return;
    const ctx = this._ctx;
    this._els.input.value = "";
    this._autosize();
    this._busy = true;
    this._els.send.disabled = true;
    this._append("user", value);
    this._renderSuggestions();

    const typing = el("div", { class: "asst-msg assistant typing" }, [
      el("div", { class: "asst-avatar", text: "VS" }),
      el("div", { class: "asst-bubble", html: "<span class='dots'><i></i><i></i><i></i></span>" }),
    ]);
    this._els.log.appendChild(typing);
    this._els.log.scrollTop = this._els.log.scrollHeight;

    try {
      const messages = this._history
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, package: ctx.package }),
      });
      const reply = await response.json();
      typing.remove();
      if (!response.ok) throw new Error(reply.error || "Assistant error");

      this._append("assistant", reply.content || "(no reply)", reply.build);

      if (reply.action === "build" && reply.build) {
        const name = String(reply.build.package_url_prefix || "").split("/").filter(Boolean).pop();
        if (name) {
          ctx.log(`Assistant built ${name}`, "ok");
          await ctx.refreshPackages();
          ctx.setPackage(name);
          ctx.goTo("viewport");
        }
      }
    } catch (error) {
      typing.remove();
      this._append("assistant", `Something went wrong: ${error.message || error}`);
    } finally {
      this._busy = false;
      this._els.send.disabled = false;
      this._els.input.focus();
    }
  },
};
