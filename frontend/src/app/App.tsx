import { useEffect } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { TitleBar } from "./TitleBar";
import { ActivityBar } from "./ActivityBar";
import { EditorTabs } from "./EditorTabs";
import { StatusBar } from "./StatusBar";
import { WorkbenchBody } from "./Workbench";
import { CommandPalette } from "./CommandPalette";
import { WelcomeTour } from "./WelcomeTour";
import { Dock } from "@/panels/dock/Dock";
import { Timeline } from "@/panels/timeline/Timeline";
import { ViewportCanvas } from "@/viewport/ViewportCanvas";
import { VerifyWorkspace } from "@/panels/verify/VerifyWorkspace";
import { LibraryWorkspace } from "@/panels/library/LibraryWorkspace";
import { TrainWorkspace } from "@/panels/train/TrainWorkspace";
import { useAppStore } from "@/state/app";
import type { WorkspaceId } from "@/state/app";
import { usePackages, useTools } from "@/api/queries";
import { registerBuiltins, registerToolCommands } from "@/commands/builtins";
import { executeCommand } from "@/commands/registry";
import { resumeRunningJobs } from "@/api/jobs";
import { engine } from "@/viewport/engine/engine";

// Virturoid Studio — the workbench shell. An IDE where the "file" is a robot:
//   TitleBar (brand · active robot · window chrome)
//   ActivityBar (icon rail: panels, explain mode, commands)
//   EditorTabs (Design / Simulate / Train / Verify / Library)
//   Editor area (3D viewport or workspace views) + resizable console panel
//   Dockable side panels: Agent (the USP) and Inspector — resizable & movable
//   StatusBar (backend · AI model · robot vitals · verification)

function EditorContent() {
  const workspace = useAppStore((s) => s.workspace);
  // Design + Simulate host the SAME engine — its canvas re-attaches, so loaded
  // content and the GL context survive tab switches.
  if (workspace === "design") return <ViewportCanvas />;
  if (workspace === "simulate") return <ViewportCanvas showSceneStrip />;
  if (workspace === "train") return <TrainWorkspace />;
  if (workspace === "verify") return <VerifyWorkspace />;
  return <LibraryWorkspace />;
}

function useGlobalHotkeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const inInput =
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement;
      if (e.ctrlKey || e.metaKey) {
        const map: Record<string, string> = {
          "1": "workspace.design",
          "2": "workspace.simulate",
          "3": "workspace.train",
          "4": "workspace.verify",
          "5": "workspace.library",
          b: "view.toggleAgent",
          i: "view.toggleInspector",
          j: "view.toggleDock",
          l: "agent.focusComposer",
        };
        const cmd = map[e.key.toLowerCase()];
        if (cmd) {
          e.preventDefault();
          void executeCommand(cmd);
        }
        return;
      }
      if (inInput) return;
      if (e.key === " ") {
        e.preventDefault();
        engine.togglePlay();
      } else if (e.key === "f" || e.key === "F") {
        engine.frameContent();
      } else if (e.key === "ArrowLeft") {
        engine.stepFrame(-1);
      } else if (e.key === "ArrowRight") {
        engine.stepFrame(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export function App() {
  const dockOpen = useAppStore((s) => s.dockOpen);
  const workspace = useAppStore((s) => s.workspace);
  const activePackage = useAppStore((s) => s.activePackage);
  const packages = usePackages();
  const tools = useTools();
  useGlobalHotkeys();

  // Boot: builtin commands + resume any jobs still running server-side.
  // Deep links: /studio?ws=verify&robot=arm_sort jumps straight to a station/robot.
  useEffect(() => {
    const dispose = registerBuiltins();
    void resumeRunningJobs();
    const params = new URLSearchParams(window.location.search);
    // First launch only: open the welcome tour (closing it dismisses forever).
    // ?tour=0 suppresses it (kiosk/testing); ?tour=1 forces it.
    if (params.get("tour") === "1" || (!useAppStore.getState().tourSeen && params.get("tour") !== "0")) {
      useAppStore.getState().setTourOpen(true);
    }
    const ws = params.get("ws");
    if (ws && ["design", "simulate", "train", "verify", "library"].includes(ws)) {
      useAppStore.getState().setWorkspace(ws as WorkspaceId);
    }
    const robot = params.get("robot");
    if (robot) useAppStore.getState().setActivePackage(robot);
    return dispose;
  }, []);

  // One palette command per backend tool, auto-registered from /api/tools.
  useEffect(() => {
    if (!tools.data) return;
    return registerToolCommands(tools.data);
  }, [tools.data]);

  // Auto-select the first robot when none is active (or the stored one vanished).
  useEffect(() => {
    const list = packages.data?.packages;
    if (!list?.length) return;
    if (!activePackage || !list.some((p) => p.id === activePackage)) {
      useAppStore.getState().setActivePackage(list[0].id);
    }
  }, [packages.data, activePackage]);

  const showTimeline = workspace === "design";
  const showDockRegion = workspace !== "library";

  return (
    <div className="flex h-full flex-col">
      <TitleBar />
      <div className="flex min-h-0 flex-1">
        <ActivityBar />
        <div className="min-w-0 flex-1">
          <WorkbenchBody>
            <div className="flex h-full min-h-0 flex-col">
              <EditorTabs />
              <div className="min-h-0 flex-1">
                <PanelGroup direction="vertical" autoSaveId={`vs.editor.v.${workspace}.v3`}>
                  <Panel id="main" order={1} minSize={35}>
                    <div className="flex h-full min-h-0 flex-col">
                      <div className="min-h-0 flex-1">
                        <EditorContent />
                      </div>
                      {showTimeline && <Timeline />}
                    </div>
                  </Panel>
                  {dockOpen && showDockRegion && (
                    <>
                      <PanelResizeHandle className="h-px bg-hairline data-[resize-handle-state=hover]:bg-accent/60" />
                      <Panel id="dock" order={2} defaultSize={26} minSize={12} maxSize={55}>
                        <Dock />
                      </Panel>
                    </>
                  )}
                </PanelGroup>
              </div>
            </div>
          </WorkbenchBody>
        </div>
      </div>
      <StatusBar />
      <CommandPalette />
      <WelcomeTour />
    </div>
  );
}
