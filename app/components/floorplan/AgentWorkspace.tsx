'use client';

// Agent mode workspace: natural-language brief -> hfagent backend
// (understand -> generate -> parse -> fix -> wall graph) -> the same
// editable CAD surface as design mode (FloorPlanEditor2D + FloorPlanView3D).

import { useState } from 'react';
import {
  Bot, Box, ChevronRight, DoorOpen, Loader2, Maximize2, MousePointer2, Ruler, Send, Square,
} from 'lucide-react';
import { roomColor } from '@/lib/plan';
import { generateAgentPlan } from '@/lib/api';
import { cn } from '@/lib/utils';
import { agentToKernelGraph, type AgentGenerateResult } from './agent-plan';
import { FloorPlanEditor2D, type EditorTool } from './FloorPlanEditor2D';
import { FloorPlanView3D } from './FloorPlanView3D';
import type { WallGraph } from './kernel';

const PRESETS = [
  {
    label: 'Community Clinic',
    brief: 'A small community clinic with 3 exam rooms, a waiting area, nurse station, and toilet.',
  },
  {
    label: 'Hospital Ward Wing',
    brief: 'A hospital ward wing with 4 patient rooms, a nurse station, storage room, and corridor.',
  },
  {
    label: 'GP Practice',
    brief: 'A GP practice with 2 exam rooms, a waiting room, reception office, and toilet.',
  },
  {
    label: 'Health Centre',
    brief: 'A medium health centre with 5 exam rooms, 2 waiting areas, a nurse station, storage, corridor, and 2 toilets.',
  },
  {
    label: 'Outpatient Dept',
    brief: 'A large hospital outpatient department with 6 exam rooms, 3 waiting areas, 2 nurse stations, a reception office, a pharmacy, a triage room, 2 corridors, storage, and 3 toilets. Waiting areas should be adjacent to exam rooms and corridors.',
  },
  {
    label: 'Inpatient Ward Floor',
    brief: `A 24-bed acute inpatient ward floor based on NHS HBN 04-01 standards, double-loaded racetrack corridor layout (~1,200 m² gross).

PATIENT ZONE — rooms lining both sides of the main corridor:
- 10 patient rooms (single-bed acute, approx 12 m² each)
- 2 patient rooms (isolation rooms at corridor ends, approx 14 m² each)
- 6 toilets (en-suite WC/shower for patient rooms, approx 5 m² each)
- 1 waiting area (family/visitor lounge at ward entrance, approx 18 m²)

NURSING CORE — between the two corridor loops:
- 2 nurse stations (central hub approx 50 m², satellite station approx 25 m²)
- 1 exam room (treatment/procedure room approx 22 m², directly adjacent to central nurse station)
- 1 exam room (medication preparation room approx 12 m², within nurse station cluster)
- 1 storage room (clean utility — sterile supplies, approx 25 m², adjacent to nurse station)
- 1 storage room (dirty utility/sluice — bedpan washer, approx 18 m², NOT adjacent to clean utility)
- 1 storage room (linen and equipment store, approx 18 m²)

STAFF ZONE — accessible only from service corridor:
- 2 corridors (main patient racetrack approx 120 m², inner staff/service corridor approx 55 m²)
- 2 offices (ward manager approx 14 m², staff rest room with kitchenette approx 20 m²)
- 2 toilets (staff WC approx 5 m² each)

ADJACENCY RULES: patient rooms open onto the main corridor only. Clean utility and dirty utility must not share a wall. Both nurse stations must adjoin the main corridor and the service corridor. Family lounge sits at the ward entrance next to the nurse reception station.`,
  },
];

export function AgentWorkspace() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentGenerateResult | null>(null);
  const [graph, setGraph] = useState<WallGraph | null>(null);
  const [view, setView] = useState<'2d' | '3d'>('2d');
  const [tool, setTool] = useState<EditorTool>('select');
  const [showDims, setShowDims] = useState(false);
  const [fitKey, setFitKey] = useState(0);

  const run = async (brief?: string) => {
    const input = (brief ?? text).trim();
    if (!input) return;
    setLoading(true);
    setError(null);
    try {
      const r = await generateAgentPlan(input);
      setResult(r);
      setGraph(agentToKernelGraph(r.wallgraph));
      setView('2d');
      setFitKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Agent generation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-1 overflow-hidden bg-background">
      {/* ── left panel: brief input + presets + result summary ── */}
      <div className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border/60 p-4">

        {/* header */}
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm font-semibold text-foreground">Plan Agent</p>
        </div>

        {/* presets */}
        <div>
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Quick Presets
          </p>
          <div className="space-y-1">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                disabled={loading}
                onClick={() => { setText(p.brief); run(p.brief); }}
                className="group flex w-full flex-col rounded-lg border border-border/60 bg-card px-3 py-2 text-left text-xs text-foreground transition-colors hover:bg-accent hover:border-border disabled:opacity-50"
              >
                <div className="flex w-full items-center justify-between">
                  <span className="font-medium">{p.label}</span>
                  <ChevronRight className="h-3 w-3 text-muted-foreground" />
                </div>
                <p className="mt-1 max-h-0 overflow-hidden text-muted-foreground transition-all duration-200 group-hover:max-h-20">
                  {p.brief}
                </p>
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="h-px flex-1 bg-border/60" />
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">or describe</span>
          <div className="h-px flex-1 bg-border/60" />
        </div>

        {/* custom brief */}
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. A ward wing with 4 patient rooms, nurse station, and corridor..."
          rows={4}
          className="w-full resize-none rounded-lg border border-border bg-background p-2.5 text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring"
        />
        <button
          onClick={() => run()}
          disabled={loading || !text.trim()}
          className="flex h-10 items-center justify-center gap-2 rounded-lg bg-primary text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          {loading ? 'Generating…' : 'Generate Plan'}
        </button>

        {loading && (
          <p className="text-xs text-muted-foreground">
            understand → generate → parse → verify → fix (up to 3 correction rounds, ~1–2 min)
          </p>
        )}

        {error && (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {/* result summary */}
        {result && (
          <div className="space-y-3 text-xs">
            <div>
              <p className="mb-1.5 font-medium uppercase tracking-wider text-muted-foreground">Program</p>
              <p className="mb-2 font-medium text-foreground">{result.program.building_type}</p>
              <div className="flex flex-wrap gap-1.5">
                {result.program.rooms.map((r) => (
                  <span
                    key={r.type}
                    className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-card px-2 py-0.5 text-foreground"
                  >
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: roomColor(r.type) }} />
                    {r.type} × {r.count}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-1 font-medium uppercase tracking-wider text-muted-foreground">Result</p>
              <p className="text-muted-foreground">
                {result.report.rounds.length} round(s) ·{' '}
                {result.report.room_count_exact
                  ? 'counts exact'
                  : result.report.final_count_exact
                    ? 'counts fixed deterministically'
                    : 'counts approximate'}{' '}
                · {result.report.pipeline}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── right panel: canvas with toolbar ── */}
      <div className="relative flex flex-1 flex-col overflow-hidden">

        {/* toolbar */}
        <div className="flex items-center gap-1 border-b border-border/60 px-3 py-2">
          <ToolBtn active={tool === 'select'} onClick={() => setTool('select')} title="Select / move">
            <MousePointer2 className="h-4 w-4" />
          </ToolBtn>
          <ToolBtn active={tool === 'door'} onClick={() => setTool('door')} title="Add door">
            <DoorOpen className="h-4 w-4" />
          </ToolBtn>
          <ToolBtn active={tool === 'passage'} onClick={() => setTool('passage')} title="Add passage">
            <Square className="h-4 w-4" />
          </ToolBtn>
          <div className="mx-1 h-5 w-px bg-border" />
          <ToolBtn active={showDims} onClick={() => setShowDims((v) => !v)} title="Wall dimensions">
            <Ruler className="h-4 w-4" />
          </ToolBtn>
          <ToolBtn active={false} onClick={() => setFitKey((k) => k + 1)} title="Fit view">
            <Maximize2 className="h-4 w-4" />
          </ToolBtn>
          <div className="ml-auto flex overflow-hidden rounded-lg border border-border">
            <button
              className={cn('px-3 py-1 text-xs', view === '2d' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}
              onClick={() => setView('2d')}
            >
              2D
            </button>
            <button
              className={cn('flex items-center gap-1 px-3 py-1 text-xs', view === '3d' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}
              onClick={() => setView('3d')}
            >
              <Box className="h-3 w-3" /> 3D
            </button>
          </div>
        </div>

        {/* canvas */}
        <div className="flex-1 overflow-hidden">
          {view === '3d' && graph ? (
            <FloorPlanView3D graph={graph} />
          ) : (
            <FloorPlanEditor2D
              graph={graph}
              boundary={[]}
              onChange={setGraph}
              tool={tool}
              showDims={showDims}
              fitSignature={fitKey}
            />
          )}
          {!graph && !loading && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <p className="text-sm text-muted-foreground">
                Pick a preset or describe a building — the agent generates an editable floor plan.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ToolBtn({
  active, onClick, title, children,
}: {
  active: boolean; onClick: () => void; title: string; children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        'flex h-8 w-8 items-center justify-center rounded-md transition-colors',
        active
          ? 'bg-primary text-primary-foreground'
          : 'text-muted-foreground hover:bg-accent hover:text-foreground',
      )}
    >
      {children}
    </button>
  );
}
