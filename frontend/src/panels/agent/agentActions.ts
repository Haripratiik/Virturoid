import { getJSON, postJSON } from "@/api/client";
import { dispatchJob } from "@/api/jobs";
import { useAssistantStore, feedId } from "@/state/assistant";
import { useAppStore, log } from "@/state/app";
import type { ChatResponse, ToolSpec } from "@/api/types";
import type { FeedEntry } from "@/state/assistant";

// The agent action bus — pure functions over the stores, so the Command Deck,
// mission cards, and the command palette all submit through ONE path.

// Build-intent detection, mirroring ui_server.py's `_fallback_build_action` so the composer and the
// server agree on what a build request looks like. Three layers:
//   1. ASKING wins: opens with an interrogative or a meta verb ("what is a quadruped?") -> not a build.
//   2. IMPERATIVE: a build verb before a robot noun ("build a quadruped robot") -> build.
//   3. NOUN PHRASE: names a robot and is not itself a question ("a quadruped robot that walks") -> build.
// Layer 3 is the point: the composer placeholder invites exactly that phrasing, and requiring an
// imperative verb made it a dead end.
const ROBOT_NOUN =
  "robots?|arms?|manipulators?|grippers?|end[ -]effectors?|mobile bases?|rovers?|drones?|" +
  "quadcopters?|bots?|machines?|quadrupeds?|bipeds?|hexapods?|octopods?|humanoids?|" +
  "walkers?|crawlers?|cobots?|exoskeletons?|gantr(?:y|ies)";
// Creature words the product does build ("build a dog", "make a spider"). Deliberately IMPERATIVE-ONLY:
// with an explicit build verb in front they are unambiguous, but as a bare noun phrase "a dog" is far
// more likely to be conversation than a build order, so layer 3 never looks at them.
const CREATURE_NOUN = "dogs?|cats?|horses?|cheetahs?|spiders?|snakes?|insects?|ants?|octopus(?:es)?";
const ASKING =
  /^\W*(?:what|why|how|when|where|which|who|whose|is|are|was|were|do|does|did|am|has|have|had|explain|tell|show|list|compare|describe|define|summari[sz]e|help|hi|hello|hey|thanks|thank)\b/i;
const BUILD_IMPERATIVE = new RegExp(
  `\\b(?:build|create|make|design|generate|assemble|spawn|construct|prototype)\\b[\\s\\S]*?\\b(?:${ROBOT_NOUN}|${CREATURE_NOUN})\\b`,
  "i",
);
const ROBOT_MENTION = new RegExp(`\\b(?:${ROBOT_NOUN})\\b`, "i");

function looksLikeBuildRequest(message: string): boolean {
  const body = message.trim();
  if (!body || ASKING.test(body)) return false;
  if (BUILD_IMPERATIVE.test(body)) return true;
  if (body.endsWith("?")) return false;
  return ROBOT_MENTION.test(body);
}

let toolsCache: ToolSpec[] | null = null;
export async function getToolSpecs(): Promise<ToolSpec[]> {
  if (!toolsCache) {
    toolsCache = (await getJSON<{ tools: ToolSpec[] }>("/api/tools")).tools;
  }
  return toolsCache;
}

async function runLightTool(name: string, args: Record<string, unknown>): Promise<void> {
  const store = useAssistantStore.getState();
  const id = feedId();
  store.push({ type: "tool", id, ts: Date.now(), tool: name, args });
  try {
    const res = await postJSON<{ ok: boolean; result?: unknown; error?: string }>("/api/tool", { tool: name, args });
    store.patch(id, { ok: res.ok, result: res.result } as Partial<FeedEntry>);
    store.push({
      type: "agent",
      id: feedId(),
      ts: Date.now(),
      text: res.ok ? JSON.stringify(res.result, null, 2).slice(0, 2000) : `Tool error: ${res.error}`,
    });
  } catch (e) {
    store.patch(id, { ok: false } as Partial<FeedEntry>);
    store.push({ type: "agent", id: feedId(), ts: Date.now(), text: `Tool call failed: ${(e as Error).message}` });
  }
}

async function dispatchBuild(prompt: string, train = false): Promise<void> {
  const store = useAssistantStore.getState();
  try {
    const job = await dispatchJob("autonomous_build", { prompt, train });
    store.push({ type: "run", id: feedId(), ts: Date.now(), jobId: job.id, goal: prompt });
  } catch (e) {
    store.push({
      type: "recovery",
      id: feedId(),
      ts: Date.now(),
      title: "Couldn't start the build",
      detail: (e as Error).message,
      actions: [{ label: "Retry", prompt }],
    });
  }
}

/** Submit one message through the agent path. Ensures the Agent panel is visible. */
export async function submitAgentMessage(raw: string): Promise<void> {
  const message = raw.trim();
  if (!message) return;
  const store = useAssistantStore.getState();
  const app = useAppStore.getState();
  if (store.busy) return;
  app.setAgentOpen(true);
  store.push({ type: "user", id: feedId(), ts: Date.now(), text: message });

  // Slash command → direct tool call (the expert path).
  if (message.startsWith("/")) {
    const [cmd, ...rest] = message.slice(1).split(" ");
    const tool = (await getToolSpecs()).find((t) => t.name === cmd);
    if (!tool) {
      store.push({ type: "agent", id: feedId(), ts: Date.now(), text: `Unknown tool "/${cmd}". Type / to see available tools.` });
      return;
    }
    const prompt = rest.join(" ").trim();
    const args: Record<string, unknown> = prompt ? { prompt, query: prompt } : {};
    if (tool.heavy) {
      const job = await dispatchJob("tool", { tool: tool.name, args });
      store.push({ type: "run", id: feedId(), ts: Date.now(), jobId: job.id, goal: `/${tool.name} ${prompt}` });
    } else {
      await runLightTool(tool.name, args);
    }
    return;
  }

  // Build path: intent → cancellable job with a live feed. Never blocks.
  if (store.mode === "build" && looksLikeBuildRequest(message)) {
    await dispatchBuild(message);
    return;
  }

  // Ask path: local-model chat; no_build guarantees the server never blocks on a build.
  store.setBusy(true);
  const pendingId = feedId();
  store.push({ type: "agent", id: pendingId, ts: Date.now(), text: "Thinking…", streaming: true });
  try {
    const history = useAssistantStore
      .getState()
      .feed.filter((e): e is Extract<FeedEntry, { type: "user" | "agent" }> => e.type === "user" || e.type === "agent")
      .slice(-12)
      .map((e) => ({ role: e.type === "user" ? "user" : "assistant", content: e.text }));
    const resp = await postJSON<ChatResponse>("/api/assistant/chat", {
      messages: history,
      no_build: true,
      model: useAssistantStore.getState().selectedModel,
    });
    if (resp.action === "build_intent" && resp.build_intent?.prompt) {
      store.patch(pendingId, { text: resp.content || "That's a build — dispatching it as a live job.", streaming: false });
      await dispatchBuild(resp.build_intent.prompt, !!resp.build_intent.train);
    } else {
      store.patch(pendingId, { text: resp.content, streaming: false });
    }
  } catch (e) {
    store.patch(pendingId, { text: `Chat failed: ${(e as Error).message}`, streaming: false });
    log("error", `Chat failed: ${(e as Error).message}`);
  } finally {
    store.setBusy(false);
  }
}
