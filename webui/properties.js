import { el, clear, pill } from "./util.js";

function numberOrNull(value) {
  return value === "" || value === null || value === undefined ? null : Number(value);
}

function renderResult(host, payload, ctx) {
  const data = payload.result || {};
  const name = String(payload.package_url_prefix || "").split("/").filter(Boolean).pop();
  const valid = Boolean(data.package_valid);
  const readiness = data.readiness || {};
  clear(host);
  host.appendChild(el("div", { class: "result-head" }, [
    pill(valid ? "valid" : "invalid", valid ? "ok" : "bad"),
    el("strong", { text: name || "" }),
    readiness.score != null ? pill(`${readiness.score}%`, readiness.ready ? "ok" : "warn") : null,
  ]));
  host.appendChild(el("div", { class: "result-meta mono" }, [
    el("div", { text: `class: ${data.selected_robot_class || "-"}` }),
    el("div", { text: `species: ${data.selected_species || "-"}` }),
    el("div", { text: `task: ${data.task_type || "-"}` }),
  ]));
}

export const propertiesPanel = {
  id: "properties",
  title: "Properties",
  dependsOnPackage: false,
  _ctx: null,

  mount(elx, ctx) {
    this._ctx = ctx;
    const tpl = document.getElementById("tpl-properties");
    clear(elx);
    const wrap = el("div", { class: "properties" });
    wrap.appendChild(el("div", { class: "panel-section-title", text: "Build a robot" }));
    wrap.appendChild(tpl.content.cloneNode(true));
    elx.appendChild(wrap);

    const form = elx.querySelector("#buildForm");
    const button = elx.querySelector("#submitButton");
    const resultEl = elx.querySelector("#buildResult");
    const promptEl = elx.querySelector("#prompt");
    const trainEl = elx.querySelector("#train");
    const outputNameEl = elx.querySelector("#outputName");

    elx.querySelector("#armExample").addEventListener("click", () => {
      promptEl.value = "Build a tabletop robot arm that sorts red and blue blocks into matching bins.";
      outputNameEl.value = "ui_arm";
    });
    elx.querySelector("#mobileExample").addEventListener("click", () => {
      promptEl.value = "Build a mobile robot base that can drive and navigate indoors to deliver parts.";
      trainEl.checked = false;
      outputNameEl.value = "ui_mobile_base";
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const c = this._ctx;
      button.disabled = true;
      button.textContent = "Compiling...";
      clear(resultEl).appendChild(el("div", { class: "empty", text: "Compiling package..." }));
      if (c) c.log("Build started...", "info");

      const body = {
        prompt: promptEl.value,
        sensor: elx.querySelector("#sensor").value || null,
        output_name: outputNameEl.value || null,
        payload_kg: numberOrNull(elx.querySelector("#payloadKg").value),
        reach_m: numberOrNull(elx.querySelector("#reachM").value),
        train: trainEl.checked,
      };

      try {
        const response = await fetch(form.dataset.endpoint || "/api/build", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Build failed");
        renderResult(resultEl, payload, c);
        const name = String(payload.package_url_prefix || "").split("/").filter(Boolean).pop();
        if (name && c) {
          c.log(`Build complete: ${name}`, "ok");
          await c.refreshPackages();
          c.setPackage(name);
          c.goTo("viewport");
        }
      } catch (error) {
        clear(resultEl).appendChild(el("div", { class: "empty error", text: `Build failed: ${error.message || error}` }));
        if (c) c.log(`Build failed: ${error.message || error}`, "bad");
      } finally {
        button.disabled = false;
        button.textContent = "Compile package";
      }
    });
  },

  onShow(ctx) { this._ctx = ctx; },
};
