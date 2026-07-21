// The command registry — the manual-parity keystone. UI buttons, the ⌘K palette
// and agent-driven actions all dispatch the SAME commands, so parity between the
// AI path and the manual path is an architectural property, not a checklist.

export interface CommandContext {
  [key: string]: unknown;
}

export interface Command {
  id: string;
  title: string;
  keywords?: string;
  section: string;
  shortcut?: string;
  when?: () => boolean;
  run: (args?: Record<string, unknown>) => void | Promise<void>;
}

const commands = new Map<string, Command>();
const listeners = new Set<() => void>();
let snapshot: Command[] = [];

function refreshSnapshot(): void {
  snapshot = [...commands.values()];
}

export function registerCommand(cmd: Command): () => void {
  commands.set(cmd.id, cmd);
  refreshSnapshot();
  for (const l of listeners) l();
  return () => {
    commands.delete(cmd.id);
    refreshSnapshot();
    for (const l of listeners) l();
  };
}

export function registerCommands(cmds: Command[]): () => void {
  const disposers = cmds.map(registerCommand);
  return () => disposers.forEach((d) => d());
}

export async function executeCommand(id: string, args?: Record<string, unknown>): Promise<boolean> {
  const cmd = commands.get(id);
  if (!cmd) return false;
  if (cmd.when && !cmd.when()) return false;
  await cmd.run(args);
  return true;
}

export function allCommands(): Command[] {
  return snapshot;
}

export function subscribeCommands(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
