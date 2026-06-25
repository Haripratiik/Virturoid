import { el, clear, pill, needsPackage, emptyState } from "./util.js";

// The Product Readiness Ledger is the single truth source for "is this package real / safe to export".
// Map its honest per-stage vocabulary to pill colors. Anything not provably real reads as a warning/blocker,
// so the panel can never look complete over a placeholder or un-run stage.
function ledgerKind(status) {
  const s = String(status || "").toLowerCase();
  if (s === "attained") return "ok";
  if (s === "not_required") return "skip";
  if (s === "scaffolded" || s === "not_run") return "warn";
  return "bad"; // placeholder, dry_run, collision, mismatch, ...
}

export const readinessSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Readiness" }),
      el("p", { text: "The Product Readiness Ledger is the truth source: a package is safe to export only when every required stage attained a real (non-placeholder) status." }),
    ]));
    this._root = el("div", {});
    panel.appendChild(this._root);
  },

  async refresh(ctx) {
    const root = clear(this._root);
    if (!ctx.package) {
      root.appendChild(needsPackage());
      return;
    }
    root.appendChild(emptyState("Loading readiness..."));

    const [ledger, mvp] = await Promise.all([
      ctx.fetchJson("reports/product_readiness_ledger.json"),
      ctx.fetchJson("reports/mvp_readiness_report.json"),
    ]);
    clear(root);

    if (!ledger && !mvp) {
      root.appendChild(emptyState("No readiness reports in this package."));
      return;
    }

    // ---- PRIMARY: the Product Readiness Ledger verdict ----
    if (ledger) {
      const safe = ledger.safe_to_export === true;
      root.appendChild(
        el("div", { class: "card score" }, [
          el("div", { class: "num", text: safe ? "SAFE" : "BLOCKED" }),
          el("div", { class: "meta" }, [
            el("div", {}, pill(safe ? "safe to export" : "not safe to export", safe ? "ok" : "bad")),
            el("p", { style: "margin:6px 0 0;", text: `Highest attained stage: ${ledger.highest_attained || "-"}` }),
          ]),
        ]),
      );

      const required = ledger.required || [];
      const stages = ledger.stages || [];
      root.appendChild(
        el("div", { class: "gate-list" },
          stages.map((s) =>
            el("div", { class: "gate" }, [
              el("div", {}, [
                pill(s.status, ledgerKind(s.status)),
                el("div", { class: "req", text: required.includes(s.stage) ? "required" : "optional" }),
              ]),
              el("div", {}, [
                el("div", { class: "label", text: s.stage }),
                el("div", { class: "detail", text: s.detail || "" }),
              ]),
            ]))),
      );

      if (Array.isArray(ledger.issues) && ledger.issues.length) {
        root.appendChild(el("h3", { style: "margin-top:18px;", text: "Honesty violations" }));
        root.appendChild(el("ul", { style: "color:var(--muted);" },
          ledger.issues.map((i) => el("li", { text: i }))));
      }
    }

    // ---- SECONDARY: legacy MVP completeness score, diagnostic only (NOT the export gate) ----
    if (mvp) {
      root.appendChild(el("h3", { style: "margin-top:22px;", text: "MVP completeness (diagnostic, not the export gate)" }));
      root.appendChild(
        el("div", { class: "card score" }, [
          el("div", { class: "num", text: `${mvp.score ?? "-"}%` }),
          el("div", { class: "meta" }, [
            el("div", {}, pill(mvp.ready ? "package complete" : "package incomplete", mvp.ready ? "ok" : "warn")),
            el("p", { style: "margin:6px 0 0;", text: "Package-completeness score. The ledger above is the export decision." }),
          ]),
        ]),
      );
      if (!ledger && Array.isArray(mvp.gates)) {
        root.appendChild(el("div", { class: "gate-list" },
          mvp.gates.map((g) =>
            el("div", { class: "gate" }, [
              el("div", {}, [pill(g.status, ledgerKind(g.status)), el("div", { class: "req", text: g.required ? "required" : "optional" })]),
              el("div", {}, [el("div", { class: "label", text: g.label || g.key }), el("div", { class: "detail", text: g.detail || "" })]),
            ]))));
      }
    }
  },
};
