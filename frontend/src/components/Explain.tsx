import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { GLOSSARY, type GlossaryKey } from "@/design/glossary";

// Hover-only help — never persistent, never covering the working surface:
//   <Tip>  — generic hover/focus tooltip (works on any element)
//   <Term> — a piece of jargon with a dotted underline + glossary definition
// Deeper onboarding lives in the dismissible WelcomeTour, not in the chrome.

export function Tip({
  text,
  children,
  side = "bottom",
  block = false,
  delay = 700,
}: {
  text: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom";
  block?: boolean;
  delay?: number;
}) {
  const [open, setOpen] = useState(false);
  const timer = useRef<number | null>(null);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  // IDE-style delay: tooltips appear only after the pointer RESTS on the
  // control, so they never flicker across the screen while mousing around.
  const show = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setOpen(true), delay);
  };
  const hide = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = null;
    setOpen(false);
    setPos(null);
  };
  useEffect(() => hide, []);

  // Portal + fixed positioning: the tooltip can never be clipped by an
  // overflow-hidden ancestor, and it is clamped inside the viewport so
  // edge-of-screen controls (e.g. the right end of the viewport toolbar)
  // still render their full tooltip.
  useLayoutEffect(() => {
    if (!open) return;
    const a = anchorRef.current?.getBoundingClientRect();
    const t = tipRef.current?.getBoundingClientRect();
    if (!a || !t) return;
    const margin = 6;
    let left = a.left + a.width / 2 - t.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - t.width - margin));
    let top = side === "bottom" ? a.bottom + margin : a.top - t.height - margin;
    if (top + t.height > window.innerHeight - margin) top = a.top - t.height - margin;
    if (top < margin) top = a.bottom + margin;
    setPos({ left, top });
  }, [open, side, text]);

  return (
    <span
      ref={anchorRef}
      className={block ? "flex min-w-0" : "inline-flex"}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {open &&
        createPortal(
          <span
            ref={tipRef}
            role="tooltip"
            style={{ position: "fixed", left: pos?.left ?? 0, top: pos?.top ?? 0, visibility: pos ? "visible" : "hidden" }}
            className="pointer-events-none z-50 w-max max-w-64 rounded-ctl border border-hairline-strong bg-overlay px-2.5 py-1.5 text-left text-2xs font-normal normal-case leading-relaxed tracking-normal text-secondary shadow-xl"
          >
            {text}
          </span>,
          document.body,
        )}
    </span>
  );
}

export function Term({ k, children }: { k: GlossaryKey; children?: ReactNode }) {
  const entry = GLOSSARY[k];
  return (
    <Tip
      text={
        <>
          <span className="mb-0.5 block font-medium text-primary">{entry.term}</span>
          {entry.short}
        </>
      }
    >
      <span className="cursor-help underline decoration-hairline-strong decoration-dotted underline-offset-2">
        {children ?? entry.term.split(" — ")[0]}
      </span>
    </Tip>
  );
}

