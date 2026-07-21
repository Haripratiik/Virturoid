import { useEffect, useState } from "react";
import { Command } from "cmdk";
import { useAppStore } from "@/state/app";
import { usePackages } from "@/api/queries";
import { allCommands, executeCommand, subscribeCommands } from "@/commands/registry";

// ⌘K — palette-complete: every registered command plus robot entity jump.

export function CommandPalette() {
  const open = useAppStore((s) => s.paletteOpen);
  const setOpen = useAppStore((s) => s.setPaletteOpen);
  const packages = usePackages();
  const [, force] = useState(0);

  useEffect(() => subscribeCommands(() => force((n) => n + 1)), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(!useAppStore.getState().paletteOpen);
      } else if (e.key === "Escape" && useAppStore.getState().paletteOpen) {
        e.preventDefault();
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setOpen]);

  if (!open) return null;

  const sections = new Map<string, ReturnType<typeof allCommands>>();
  for (const cmd of allCommands()) {
    if (cmd.when && !cmd.when()) continue;
    if (!sections.has(cmd.section)) sections.set(cmd.section, []);
    sections.get(cmd.section)!.push(cmd);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
      onClick={() => setOpen(false)}
      role="presentation"
    >
      <div onClick={(e) => e.stopPropagation()} role="presentation" className="w-[min(600px,90vw)]">
        <Command
          label="Command palette"
          className="overflow-hidden rounded-panel border border-hairline-strong bg-raised shadow-2xl"
        >
          <Command.Input
            autoFocus
            placeholder="Type a command or robot name…"
            className="w-full border-b border-hairline bg-transparent px-4 py-3 text-sm text-primary outline-none placeholder:text-muted"
          />
          <Command.List className="max-h-80 overflow-y-auto p-1.5">
            <Command.Empty className="px-3 py-6 text-center text-xs text-muted">No results.</Command.Empty>
            {[...sections.entries()].map(([section, cmds]) => (
              <Command.Group
                key={section}
                heading={section}
                className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 [&_[cmdk-group-heading]]:text-2xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted"
              >
                {cmds.map((cmd) => (
                  <Command.Item
                    key={cmd.id}
                    value={`${cmd.title} ${cmd.keywords ?? ""}`}
                    onSelect={() => {
                      setOpen(false);
                      void executeCommand(cmd.id);
                    }}
                    className="flex cursor-pointer items-center justify-between gap-3 rounded-ctl px-2.5 py-1.5 text-xs text-secondary data-[selected=true]:bg-accent-dim data-[selected=true]:text-accent"
                  >
                    <span>{cmd.title}</span>
                    {cmd.shortcut && <kbd className="font-mono text-2xs text-muted">{cmd.shortcut}</kbd>}
                  </Command.Item>
                ))}
              </Command.Group>
            ))}
            <Command.Group
              heading="Robots"
              className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 [&_[cmdk-group-heading]]:text-2xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted"
            >
              {(packages.data?.packages ?? []).map((p) => (
                <Command.Item
                  key={p.id}
                  value={`robot ${p.id} ${p.robot_class ?? ""}`}
                  onSelect={() => {
                    setOpen(false);
                    useAppStore.getState().setActivePackage(p.id);
                  }}
                  className="flex cursor-pointer items-center justify-between gap-3 rounded-ctl px-2.5 py-1.5 text-xs text-secondary data-[selected=true]:bg-accent-dim data-[selected=true]:text-accent"
                >
                  <span className="font-mono">{p.id}</span>
                  <span className="text-2xs text-muted">{p.robot_class ?? ""}</span>
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
