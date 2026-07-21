import { useEffect, useState } from "react";
import { X, ArrowRight, ArrowLeft, Bot, SlidersHorizontal, ShieldCheck, Hexagon } from "lucide-react";
import { useAppStore } from "@/state/app";

// The welcome tour — a Cursor-style intro shown ONCE on first launch and
// otherwise never in the way. Reopen from the graduation-cap icon in the
// activity bar. Closing it (X, Esc, Skip, or finishing) dismisses it forever.

type Region = "tabs" | "viewport" | "agent" | "inspector" | "console" | null;

// A miniature of the actual workbench so newcomers can map words to places.
function MiniMap({ highlight }: { highlight: Region }) {
  const cls = (r: Region, base: string) =>
    `${base} rounded-[3px] border transition-colors ${
      highlight === r
        ? "border-accent/70 bg-accent-dim text-accent"
        : "border-hairline bg-raised/40 text-muted"
    }`;
  return (
    <div className="flex h-40 gap-1 rounded-card border border-hairline bg-canvas p-2" aria-hidden>
      <div className="flex w-5 flex-col items-center gap-1 rounded-[3px] border border-hairline bg-raised/40 py-1">
        <span className="h-1.5 w-1.5 rounded-full bg-muted/50" />
        <span className="h-1.5 w-1.5 rounded-full bg-muted/50" />
        <span className="h-1.5 w-1.5 rounded-full bg-muted/50" />
      </div>
      <div className={cls("inspector", "flex w-1/5 items-center justify-center text-2xs")}>Inspector</div>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className={cls("tabs", "flex h-5 items-center justify-center text-2xs")}>Design · Simulate · Train · Verify</div>
        <div className={cls("viewport", "flex flex-1 items-center justify-center text-2xs")}>3D viewport</div>
        <div className={cls("console", "flex h-8 items-center justify-center text-2xs")}>Console · Jobs · Files</div>
      </div>
      <div className={cls("agent", "flex w-1/4 items-center justify-center text-2xs")}>Agent</div>
    </div>
  );
}

const STEPS: Array<{
  title: string;
  body: React.ReactNode;
  region: Region;
  icon: React.ReactNode;
}> = [
  {
    title: "Welcome to Virturoid Studio",
    icon: <Hexagon size={15} className="text-accent" aria-hidden />,
    region: "viewport",
    body: (
      <>
        This is a workbench for making real robots: describe one, watch it get designed and tested in genuine
        physics, then take the files and build it. It is laid out like a code editor — except instead of a
        script, the thing in the middle is your robot.
      </>
    ),
  },
  {
    title: "The Agent does the heavy lifting",
    icon: <Bot size={15} className="text-agent" aria-hidden />,
    region: "agent",
    body: (
      <>
        Tell the Agent what you want in plain language — <em>“build a robot arm that sorts blocks”</em>. It
        designs, simulates and reports back as a live job you can watch and cancel. <strong>Build</strong> mode
        makes things; <strong>Ask</strong> mode only answers questions. Anything shown in blue was authored by
        the AI.
      </>
    ),
  },
  {
    title: "Everything is also manual",
    icon: <SlidersHorizontal size={15} className="text-secondary" aria-hidden />,
    region: "inspector",
    body: (
      <>
        The Inspector holds the robot's properties, skeleton, parts list and sensors — click any part in 3D to
        jump to it. The viewport toolbar has camera presets, grid, wireframe and screenshots. Press{" "}
        <kbd className="rounded border border-hairline bg-raised px-1 font-mono text-2xs">Ctrl+K</kbd> to search
        every command. Panels drag to resize and can be moved to either side.
      </>
    ),
  },
  {
    title: "Honest by design",
    icon: <ShieldCheck size={15} className="text-ok" aria-hidden />,
    region: "tabs",
    body: (
      <>
        Robotics is full of demos that only work once. The <strong>Verify</strong> tab separates what has been
        physically proven in simulation from what is only claimed — nothing exports until its gates pass. The
        little dots on the tabs are that same truth, live.
      </>
    ),
  },
];

export function WelcomeTour() {
  const open = useAppStore((s) => s.tourOpen);
  const setTourOpen = useAppStore((s) => s.setTourOpen);
  const [step, setStep] = useState(0);

  // Reset to the first page whenever the tour opens; Esc closes.
  useEffect(() => {
    if (!open) return;
    setStep(0);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setTourOpen(false);
      if (e.key === "ArrowRight") setStep((s) => Math.min(s + 1, STEPS.length - 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setTourOpen]);

  if (!open) return null;
  const current = STEPS[step];
  const last = step === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" role="dialog" aria-modal="true" aria-label="Welcome tour">
      <div className="w-[min(520px,94vw)] overflow-hidden rounded-panel border border-hairline-strong bg-panel shadow-2xl">
        <div className="flex items-center justify-between border-b border-hairline px-4 py-2.5">
          <div className="flex items-center gap-2">
            {current.icon}
            <span className="text-sm font-semibold text-primary">{current.title}</span>
          </div>
          <button
            type="button"
            aria-label="Skip the tour"
            onClick={() => setTourOpen(false)}
            className="rounded-ctl p-1 text-muted hover:bg-raised hover:text-primary"
          >
            <X size={14} aria-hidden />
          </button>
        </div>

        <div className="flex flex-col gap-3 px-4 py-3">
          <MiniMap highlight={current.region} />
          <p className="text-xs leading-relaxed text-secondary">{current.body}</p>
        </div>

        <div className="flex items-center justify-between border-t border-hairline px-4 py-2.5">
          <button
            type="button"
            onClick={() => setTourOpen(false)}
            className="text-2xs text-muted hover:text-secondary"
          >
            Skip — I know my way around
          </button>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5" aria-label={`Step ${step + 1} of ${STEPS.length}`}>
              {STEPS.map((s, i) => (
                <button
                  key={s.title}
                  type="button"
                  aria-label={`Go to step ${i + 1}`}
                  onClick={() => setStep(i)}
                  className={`h-1.5 rounded-full transition-all ${i === step ? "w-4 bg-accent" : "w-1.5 bg-hairline-strong hover:bg-muted"}`}
                />
              ))}
            </div>
            {step > 0 && (
              <button
                type="button"
                onClick={() => setStep(step - 1)}
                className="flex items-center gap-1 rounded-ctl border border-hairline px-2.5 py-1 text-xs text-secondary hover:text-primary"
              >
                <ArrowLeft size={12} aria-hidden />
                Back
              </button>
            )}
            <button
              type="button"
              onClick={() => (last ? setTourOpen(false) : setStep(step + 1))}
              className="flex items-center gap-1 rounded-ctl border border-accent/40 bg-accent-dim px-2.5 py-1 text-xs font-medium text-accent hover:bg-accent/25"
            >
              {last ? "Start building" : "Next"}
              {!last && <ArrowRight size={12} aria-hidden />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
