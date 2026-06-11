'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  MousePointer2,
  DoorOpen,
  SeparatorHorizontal,
  Magnet,
  Compass,
  Spline,
  Maximize2,
  Box,
  Square,
  Play,
  Loader2,
  GripVertical,
} from 'lucide-react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import { FloorPlanEditor2D, type EditorTool } from './FloorPlanEditor2D';
import { FloorPlanView3D } from './FloorPlanView3D';
import { BOUNDARY_SAMPLES } from '@/lib/boundary-samples';
import { planToGraph, type WallGraph } from './kernel';
import { cn } from '@/lib/utils';
import type { DatasetId } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';
import type { Plan, Pt } from '@/lib/plan';

interface Props {
  graph: GeneratedGraph;
  dataset: DatasetId;
  plan: Plan | null;
  loading: boolean;
  error: string | null;
  fitKey: number;
  onGraphChange: (g: GeneratedGraph) => void;
  onGenerate: (boundary: Pt[], axisAngle: number | null) => void;
  onPlanChange: (plan: Plan) => void;
}

export function DesignWorkspace({
  graph,
  dataset,
  plan,
  loading,
  error,
  fitKey,
  onGraphChange,
  onGenerate,
}: Props) {
  const [boundary, setBoundary] = useState<Pt[]>(BOUNDARY_SAMPLES[0].pts);
  const [tool, setTool] = useState<EditorTool>('select');
  const [gridSnap, setGridSnap] = useState(false);
  const [axisAuto, setAxisAuto] = useState(true);
  const [axisDeg, setAxisDeg] = useState(0);
  const [view, setView] = useState<'2d' | '3d'>('2d');
  const [localFit, setLocalFit] = useState(0);
  const [leftPct, setLeftPct] = useState(40);
  const [graphModel, setGraphModel] = useState<WallGraph | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const draggingSplit = useRef(false);

  // Build the editable wall graph from a freshly generated plan (fitKey bumps
  // per generation). Edits then mutate the graph; regeneration rebuilds it.
  const lastBuilt = useRef<number | null>(null);
  useEffect(() => {
    if (plan && lastBuilt.current !== fitKey) {
      lastBuilt.current = fitKey;
      setGraphModel(planToGraph(plan));
    }
  }, [fitKey, plan]);

  const fitSignature = useMemo(() => fitKey * 1000 + localFit, [fitKey, localFit]);

  const generate = useCallback(() => {
    onGenerate(boundary, axisAuto ? null : axisDeg);
  }, [boundary, axisAuto, axisDeg, onGenerate]);

  const onSplitMove = useCallback((e: React.PointerEvent) => {
    if (!draggingSplit.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftPct(Math.max(16, Math.min(84, pct)));
  }, []);

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden bg-background">
      {/* ── Left: bubble graph (input) ───────────────────────────────── */}
      <section className="flex min-w-0 flex-col border-r border-border/60 bg-card/40" style={{ width: `${leftPct}%` }}>
        <PaneHeader label="Bubble Graph" hint="select from history · or draw here" />
        <div className="relative flex-1 overflow-hidden">
          <BubbleGraphCanvas graph={graph} dataset={dataset} mode="edit" onChange={onGraphChange} />
        </div>
      </section>

      {/* ── Splitter ─────────────────────────────────────────────────── */}
      <div
        onPointerDown={(e) => {
          draggingSplit.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={onSplitMove}
        onPointerUp={(e) => {
          draggingSplit.current = false;
          try {
            e.currentTarget.releasePointerCapture(e.pointerId);
          } catch {
            /* noop */
          }
        }}
        onDoubleClick={() => setLeftPct(40)}
        title="Drag to resize · double-click to reset"
        className="group flex w-1.5 shrink-0 cursor-col-resize items-center justify-center bg-border/40 transition-colors hover:bg-blue-500/60"
      >
        <GripVertical className="h-4 w-4 text-muted-foreground/50 transition-colors group-hover:text-white" />
      </div>

      {/* ── Right: CAD plan (output, editable) ───────────────────────── */}
      <section className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex flex-wrap items-center gap-1.5 border-b border-border/60 bg-card/60 px-3 py-2 backdrop-blur">
          <ToolGroup>
            <ToolButton active={tool === 'select'} onClick={() => setTool('select')} title="Select / move walls, rooms, openings">
              <MousePointer2 className="h-4 w-4" />
            </ToolButton>
            <ToolButton active={tool === 'door'} onClick={() => setTool('door')} title="Add door (click a wall)">
              <DoorOpen className="h-4 w-4" />
            </ToolButton>
            <ToolButton active={tool === 'passage'} onClick={() => setTool('passage')} title="Add passage (click a wall)">
              <SeparatorHorizontal className="h-4 w-4" />
            </ToolButton>
            <ToolButton active={tool === 'boundary'} onClick={() => setTool((t) => (t === 'boundary' ? 'select' : 'boundary'))} title="Draw / edit boundary on canvas">
              <Spline className="h-4 w-4" />
            </ToolButton>
          </ToolGroup>

          <Divider />

          <ToolButton active={gridSnap} onClick={() => setGridSnap((s) => !s)} title="Snap to grid">
            <Magnet className="h-4 w-4" />
          </ToolButton>

          <div className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-background/60 px-2 py-1">
            <Compass className="h-3.5 w-3.5 text-muted-foreground" />
            <button
              className={cn(
                'rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors',
                axisAuto ? 'bg-foreground text-background' : 'text-muted-foreground hover:text-foreground',
              )}
              onClick={() => setAxisAuto((a) => !a)}
              title="Auto-detect building axis from boundary"
            >
              auto
            </button>
            {!axisAuto && (
              <div className="flex items-center gap-1">
                <input type="range" min={0} max={89} value={axisDeg} onChange={(e) => setAxisDeg(Number(e.target.value))} className="h-1 w-20 accent-blue-600" />
                <span className="w-8 text-right font-mono text-[11px] tabular-nums text-foreground">{axisDeg}°</span>
              </div>
            )}
          </div>

          <ToolButton onClick={() => setLocalFit((f) => f + 1)} title="Fit to view">
            <Maximize2 className="h-4 w-4" />
          </ToolButton>

          <div className="ml-auto flex items-center gap-1.5">
            <ToolGroup>
              <ToolButton active={view === '2d'} onClick={() => setView('2d')} title="2D plan">
                <Square className="h-4 w-4" />
              </ToolButton>
              <ToolButton active={view === '3d'} onClick={() => setView('3d')} title="3D view">
                <Box className="h-4 w-4" />
              </ToolButton>
            </ToolGroup>

            <button
              onClick={generate}
              disabled={loading}
              className={cn(
                'flex items-center gap-2 rounded-lg bg-blue-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm',
                'transition-all duration-200 hover:bg-blue-500 hover:shadow active:scale-[0.98] disabled:opacity-60',
              )}
            >
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {plan ? 'Update Plan' : 'Generate Plan'}
            </button>
          </div>
        </div>

        {/* boundary samples strip (when boundary tool is active) */}
        {tool === 'boundary' && (
          <div className="flex flex-wrap items-center gap-1.5 border-b border-border/60 bg-card/70 px-3 py-2 [animation:fpFadeSlide_160ms_ease-out]">
            <span className="mr-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Samples</span>
            {BOUNDARY_SAMPLES.map((s) => (
              <button
                key={s.name}
                onClick={() => setBoundary(s.pts)}
                className="rounded-md border border-border/70 bg-background px-2 py-1 text-[11px] text-foreground transition-colors hover:border-blue-500 hover:text-blue-600"
              >
                {s.name}
              </button>
            ))}
          </div>
        )}

        {/* canvas */}
        <div className="relative flex-1 overflow-hidden">
          {error ? (
            <Centered>
              <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">{error}</div>
            </Centered>
          ) : view === '3d' && graphModel ? (
            <FloorPlanView3D graph={graphModel} />
          ) : (
            <>
              <FloorPlanEditor2D
                graph={graphModel}
                boundary={boundary}
                onChange={setGraphModel}
                onBoundaryChange={setBoundary}
                gridSnap={gridSnap}
                tool={tool}
                fitSignature={fitSignature}
              />
              {!graphModel && !loading && (
                <div className="pointer-events-none absolute inset-x-0 bottom-6 flex justify-center">
                  <div className="pointer-events-auto rounded-full border border-border/70 bg-card/95 px-4 py-2 text-xs text-muted-foreground shadow-md backdrop-blur">
                    Adjust the boundary &amp; graph, then <span className="font-medium text-foreground">Generate Plan</span>
                  </div>
                </div>
              )}
              {loading && (
                <div className="pointer-events-none absolute right-4 top-4 flex items-center gap-2 rounded-full border border-border/70 bg-card/95 px-3 py-1.5 text-xs text-muted-foreground shadow-md backdrop-blur">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> laying out…
                </div>
              )}
            </>
          )}
        </div>

        <StatusBar graph={graphModel} boundary={boundary} gridSnap={gridSnap} tool={tool} view={view} />
      </section>
    </div>
  );
}

function PaneHeader({ label, hint }: { label: string; hint: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-border/60 px-3 py-2">
      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-foreground/80">{label}</span>
      <span className="text-[10px] text-muted-foreground">{hint}</span>
    </div>
  );
}

function ToolGroup({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-0.5 rounded-lg border border-border/60 bg-background/60 p-0.5">{children}</div>;
}

function ToolButton({
  children,
  active,
  onClick,
  title,
}: {
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        'flex h-7 w-7 items-center justify-center rounded-md transition-all duration-150',
        active ? 'bg-foreground text-background shadow-sm' : 'text-muted-foreground hover:bg-muted hover:text-foreground',
      )}
    >
      {children}
    </button>
  );
}

function Divider() {
  return <div className="mx-0.5 h-5 w-px bg-border/70" />;
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full w-full items-center justify-center p-6">{children}</div>;
}

function StatusBar({
  graph,
  boundary,
  gridSnap,
  tool,
  view,
}: {
  graph: WallGraph | null;
  boundary: Pt[];
  gridSnap: boolean;
  tool: EditorTool;
  view: '2d' | '3d';
}) {
  const area = useMemo(() => polyArea(boundary), [boundary]);
  return (
    <div className="flex items-center gap-4 border-t border-border/60 bg-card/60 px-3 py-1 font-mono text-[10px] tabular-nums text-muted-foreground backdrop-blur">
      <span>{view.toUpperCase()}</span>
      <span>tool: {tool}</span>
      <span>snap: {gridSnap ? 'on' : 'off'}</span>
      <span className="ml-auto">boundary ≈ {area.toFixed(1)} m²</span>
      {graph && (
        <>
          <span>{graph.rooms.length} rooms</span>
          <span>{graph.walls.length} walls</span>
          <span>{graph.openings.length} openings</span>
          <span>axis {graph.grid.angleDeg.toFixed(1)}°</span>
        </>
      )}
    </div>
  );
}

function polyArea(pts: Pt[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}
