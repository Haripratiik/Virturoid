// Small dependency-free DOM + fetch helpers shared by section modules.

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") {
      for (const [dk, dv] of Object.entries(value)) node.dataset[dk] = dv;
    } else {
      node.setAttribute(key, value);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

export function pill(text, kind = "skip") {
  return el("span", { class: `pill ${kind}`, text: String(text) });
}

export function statusKind(status) {
  const value = String(status || "").toLowerCase();
  if (["pass", "ok", "success", "valid", "available", "true"].includes(value)) return "ok";
  if (["fail", "failed", "error", "missing", "invalid", "false"].includes(value)) return "bad";
  if (["skip", "skipped", "not_checked", "not_required"].includes(value)) return "skip";
  return "warn";
}

export function packageUrl(packageId, relativePath) {
  const file = String(relativePath || "")
    .replace(/\\/g, "/")
    .replace(/^\/+/, "")
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  const pkg = String(packageId || "")
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  return `/package/${pkg}/${file}`;
}

export async function getJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn("getJson failed", url, error);
    return null;
  }
}

export async function getText(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.text();
  } catch (error) {
    console.warn("getText failed", url, error);
    return null;
  }
}

export function emptyState(message, isError = false) {
  return el("div", { class: `empty${isError ? " error" : ""}`, text: message });
}

export function needsPackage() {
  return emptyState("No package selected. Build one in the Build section first.");
}
