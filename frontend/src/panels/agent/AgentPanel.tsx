import { useEffect, useRef, useState } from "react";
import {
  Bot,
  User,
  Wrench,
  CornerDownLeft,
  Sparkles,
  MessageCircleQuestion,
  Trash2,
} from "lucide-react";
import { useAssistantStore, feedId } from "@/state/assistant";
import { useAppStore } from "@/state/app";
import { useTools, useInvalidateAfterBuild } from "@/api/queries";
import { onJobFinished } from "@/api/jobs";
import { submitAgentMessage } from "./agentActions";
import { RunCard } from "./RunCard";
import { Tip } from "@/components/Explain";
import type { Job } from "@/api/types";
import type { FeedEntry } from "@/state/assistant";

// The Agent panel — a first-class side panel, like the AI pane in a modern
// IDE. Conversation on top, composer pinned to the bottom. Builds run as
// live, cancellable jobs inline in the feed. Ion blue marks AI-authored work.

const MISSIONS = [
  { title: "Sorting arm", prompt: "Build a tabletop robot arm that sorts red and blue blocks into matching bins" },
  { title: "Walking quadruped", prompt: "Build a quadruped robot that walks" },
  { title: "Delivery rover", prompt: "Build a mobile base that delivers parts indoors" },
  { title: "Six-legged walker", prompt: "Build a six-legged walking robot" },
];

function FeedItem({ entry }: { entry: FeedEntry }) {
  switch (entry.type) {
    case "user":
      return (
        <div className="flex gap-2">
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-ctl bg-raised text-secondary">
            <User size={11} aria-label="you" />
          </span>
          <div className="min-w-0 whitespace-pre-wrap text-xs leading-relaxed text-primary">{entry.text}</div>
        </div>
      );
    case "agent":
      return (
        <div className="flex gap-2">
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-ctl bg-agent-dim text-agent">
            <Bot size={11} aria-label="agent" />
          </span>
          <div className={`min-w-0 whitespace-pre-wrap text-xs leading-relaxed text-secondary ${entry.streaming ? "vs-streaming" : ""}`}>
            {entry.text}
          </div>
        </div>
      );
    case "run":
      return <RunCard jobId={entry.jobId} goal={entry.goal} />;
    case "recovery":
      return (
        <div className="rounded-card border border-warn/30 bg-warn-dim/40 p-2.5">
          <div className="mb-1 text-xs font-medium text-warn">{entry.title}</div>
          <div className="mb-2 text-xs text-secondary">{entry.detail}</div>
          <div className="flex flex-wrap gap-1.5">
            {entry.actions.map((a) => (
              <button
                key={a.label}
                type="button"
                onClick={() => void submitAgentMessage(a.prompt)}
                className="rounded-ctl border border-hairline bg-panel px-2 py-0.5 text-2xs text-secondary hover:text-primary"
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
      );
    case "tool":
      return (
        <div className="flex items-center gap-2 rounded-ctl bg-raised/50 px-2 py-1 font-mono text-2xs text-secondary">
          <Wrench size={11} className="shrink-0 text-agent" aria-hidden />
          <span className="truncate">
            {entry.tool}({JSON.stringify(entry.args)}) {entry.ok === undefined ? "…" : entry.ok ? "✓" : "✗"}
          </span>
        </div>
      );
  }
}

function EmptyFeed() {
  return (
    <div className="flex h-full flex-col justify-center gap-4 p-4">
      <div>
        <div className="mb-1 text-sm font-medium text-primary">What should we build?</div>
        <div className="text-xs leading-relaxed text-muted">
          Describe any robot in plain language. The agent designs it, tests it in physics, and reports honestly
          what works. Try one:
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        {MISSIONS.map((m) => (
          <button
            key={m.title}
            type="button"
            onClick={() => void submitAgentMessage(m.prompt)}
            className="group rounded-card border border-hairline bg-raised/40 p-2.5 text-left transition-colors hover:border-agent/50"
          >
            <div className="mb-0.5 flex items-center gap-1.5 text-xs font-medium text-primary group-hover:text-agent">
              <Sparkles size={11} className="text-agent" aria-hidden />
              {m.title}
            </div>
            <div className="text-2xs leading-relaxed text-muted">{m.prompt}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

export function AgentPanel() {
  const mode = useAssistantStore((s) => s.mode);
  const busy = useAssistantStore((s) => s.busy);
  const feed = useAssistantStore((s) => s.feed);
  const tools = useTools();
  const invalidate = useInvalidateAfterBuild();
  const [text, setText] = useState("");
  const [showTools, setShowTools] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [feed]);

  // Build completion → refresh caches, load the robot, summarize, route failures.
  useEffect(() => {
    return onJobFinished((job: Job) => {
      invalidate();
      const store = useAssistantStore.getState();
      const result = (job.result ?? {}) as Record<string, unknown>;
      if (job.status === "succeeded" && job.kind === "autonomous_build") {
        if (result.requires_clarification === true) {
          store.push({
            type: "recovery",
            id: feedId(),
            ts: Date.now(),
            title: "Need a clearer robot brief",
            detail:
              typeof result.clarification === "string"
                ? result.clarification
                : "Specify a body plan and task before Virturoid generates a robot.",
            actions: [
              { label: "Build a rover", prompt: "Build a wheeled rover that navigates to a goal" },
              { label: "Build an arm", prompt: "Build a tabletop robot arm that sorts blocks" },
            ],
          });
          return;
        }
        const name = typeof result.output_name === "string" ? result.output_name : null;
        if (name) useAppStore.getState().setActivePackage(name);
        const cls = result.robot_class ?? "robot";
        const sr =
          typeof result.final_success_rate === "number"
            ? ` Task success ${Math.round((result.final_success_rate as number) * 100)}%.`
            : "";
        store.push({
          type: "agent",
          id: feedId(),
          ts: Date.now(),
          text: `Build finished: a ${cls}${result.species ? ` (${result.species})` : ""} is on the stage.${sr} The Verify tab holds the honesty gates.`,
        });
        if (typeof result.species_note === "string" && result.species_exact === false) {
          store.push({
            type: "recovery",
            id: feedId(),
            ts: Date.now(),
            title: "Built the nearest buildable relative",
            detail: result.species_note,
            actions: [{ label: "Ask why", prompt: "Why couldn't the requested species be built exactly?" }],
          });
        }
      } else if (job.status === "failed") {
        store.push({
          type: "recovery",
          id: feedId(),
          ts: Date.now(),
          title: "The build failed",
          detail: job.error || "The pipeline raised an error.",
          actions: [
            { label: "Retry", prompt: String(job.args?.prompt ?? "") },
            { label: "Simplify the request", prompt: "Build a tabletop robot arm that sorts blocks" },
          ],
        });
      } else if (job.status === "cancelled") {
        store.push({
          type: "agent",
          id: feedId(),
          ts: Date.now(),
          text: "Build cancelled. Partial artifacts stay on disk (honest state).",
        });
      }
    });
  }, [invalidate]);

  async function send() {
    const message = text.trim();
    if (!message || busy) return;
    setText("");
    setShowTools(false);
    await submitAgentMessage(message);
  }

  const filteredTools = (tools.data ?? []).filter((t) => text.length < 2 || t.name.includes(text.slice(1)));

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Conversation */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        {feed.length === 0 ? (
          <EmptyFeed />
        ) : (
          <div className="flex flex-col gap-3 p-3">
            {feed.map((entry) => (
              <FeedItem key={entry.id} entry={entry} />
            ))}
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="relative shrink-0 border-t border-hairline p-2">
        {showTools && filteredTools.length > 0 && (
          <div className="absolute bottom-full left-2 right-2 z-40 mb-1 max-h-56 overflow-y-auto rounded-card border border-hairline bg-raised p-1 shadow-lg">
            {filteredTools.map((t) => (
              <button
                key={t.name}
                type="button"
                onClick={() => {
                  setText(`/${t.name} `);
                  setShowTools(false);
                  inputRef.current?.focus();
                }}
                className="flex w-full items-start gap-2 rounded-ctl px-2 py-1.5 text-left hover:bg-overlay"
              >
                <span className="font-mono text-2xs text-agent">/{t.name}</span>
                <span className="min-w-0 flex-1 truncate text-2xs text-muted">
                  {t.description}
                  {t.heavy ? " (slow, real physics)" : ""}
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="rounded-card border border-hairline bg-canvas focus-within:border-agent/60">
          <textarea
            ref={inputRef}
            id="agent-composer"
            value={text}
            rows={Math.min(6, Math.max(2, text.split("\n").length))}
            placeholder={
              mode === "build"
                ? "Describe a robot to build…"
                : "Ask about the active robot — nothing will be changed…"
            }
            aria-label="Message the agent"
            className="w-full resize-none bg-transparent px-2.5 pt-2 text-xs leading-relaxed text-primary outline-none placeholder:text-muted"
            onChange={(e) => {
              setText(e.target.value);
              setShowTools(e.target.value.startsWith("/"));
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
              if (e.key === "Escape") setShowTools(false);
            }}
          />
          <div className="flex items-center gap-1 px-1.5 pb-1.5">
            {/* Mode toggle */}
            <div className="flex overflow-hidden rounded-ctl border border-hairline" role="radiogroup" aria-label="Agent mode">
              {(
                [
                  { id: "build", label: "Build", Icon: Sparkles, hint: "Build mode — describing a robot starts a real, cancellable build job" },
                  { id: "ask", label: "Ask", Icon: MessageCircleQuestion, hint: "Ask mode — questions and explanations only, nothing is ever changed" },
                ] as const
              ).map(({ id, label, Icon, hint }) => (
                <Tip key={id} text={hint} side="top">
                  <button
                    type="button"
                    role="radio"
                    aria-checked={mode === id}
                    onClick={() => useAssistantStore.getState().setMode(id)}
                    className={`flex items-center gap-1 px-2 py-1 text-2xs font-medium ${
                      mode === id ? "bg-agent-dim text-agent" : "text-muted hover:text-secondary"
                    }`}
                  >
                    <Icon size={10} aria-hidden />
                    {label}
                  </button>
                </Tip>
              ))}
            </div>
            <Tip text="Type / to call any backend tool directly — the expert path" side="top">
              <span className="px-1 font-mono text-2xs text-muted">/</span>
            </Tip>
            <span className="flex-1" />
            {feed.length > 0 && (
              <Tip text="Clear this conversation (robots and files are untouched)" side="top">
                <button
                  type="button"
                  aria-label="Clear conversation"
                  onClick={() => useAssistantStore.getState().clear()}
                  className="rounded-ctl p-1 text-muted hover:bg-raised hover:text-primary"
                >
                  <Trash2 size={12} aria-hidden />
                </button>
              </Tip>
            )}
            <button
              type="button"
              onClick={() => void send()}
              disabled={!text.trim() || busy}
              title="Send (Enter)"
              aria-label="Send"
              className="flex h-6 w-6 items-center justify-center rounded-ctl bg-agent-dim text-agent disabled:opacity-40"
            >
              <CornerDownLeft size={12} aria-hidden />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
