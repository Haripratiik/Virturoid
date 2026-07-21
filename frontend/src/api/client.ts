// Thin fetch layer. All endpoints are same-origin (Vite dev-proxies /api + /package
// to the Python server; in prod the Python server serves the app itself).

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText} for ${url}`);
  return (await res.json()) as T;
}

/** Like getJSON but resolves null on 404 / network failure — for optional package artifacts. */
export async function tryJSON<T>(url: string): Promise<T | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText} for ${url}`);
  return (await res.json()) as T;
}

export async function tryText(url: string): Promise<string | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

export async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = (await res.json()) as { error?: string };
      if (data?.error) detail = data.error;
    } catch {
      /* keep status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function packageUrl(pkg: string, rel: string): string {
  const clean = rel.replace(/^\/+/, "");
  return `/package/${encodeURIComponent(pkg)}/${clean.split("/").map(encodeURIComponent).join("/")}`;
}
