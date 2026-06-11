'use client';

import { useCallback, useMemo, useState } from 'react';
import { HelpCircle, Search, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import { retrieveGraph } from '@/lib/api';
import {
  RETRIEVAL_DATASET,
  RETRIEVAL_MODES,
  ROOM_TYPES_MSD,
  type RetrievalMode,
} from '@/lib/constants';
import type { GeneratedGraph, RetrievedItem } from '@/lib/types';
import { cn } from '@/lib/utils';


const EMPTY_QUERY: GeneratedGraph = {
  num_nodes: 0,
  num_edges: 0,
  rooms: [],
  adjacency: [],
  edge_types: [],
  nodes: [],
  edges: [],
};

const DEFAULT_MODE: RetrievalMode = RETRIEVAL_MODES[0].id;


interface SelectedMatch {
  item: RetrievedItem;
  rank: number;
}


export function RetrievePanel() {
  const [queryGraph, setQueryGraph] = useState<GeneratedGraph>(EMPTY_QUERY);
  const [subMode, setSubMode] = useState<RetrievalMode>(DEFAULT_MODE);
  const [k, setK] = useState(5);
  const [results, setResults] = useState<RetrievedItem[]>([]);
  const [elapsedSec, setElapsedSec] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<SelectedMatch | null>(null);

  const queryHasNodes = queryGraph.nodes.length > 0;
  const activeModeMeta = useMemo(
    () => RETRIEVAL_MODES.find((m) => m.id === subMode) ?? RETRIEVAL_MODES[0],
    [subMode],
  );

  const handleRetrieve = useCallback(async () => {
    if (!queryHasNodes) {
      setError('Add at least one room to the query graph.');
      return;
    }
    setLoading(true);
    setError(null);
    setSelectedMatch(null);
    try {
      const res = await retrieveGraph(queryGraph, subMode, k);
      setResults(res.results);
      setElapsedSec(res.retrieval_seconds);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Retrieval failed');
      setResults([]);
      setElapsedSec(null);
    } finally {
      setLoading(false);
    }
  }, [queryGraph, queryHasNodes, subMode, k]);

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Top control bar: algorithm select, k, retrieve action */}
      <TooltipProvider>
      <div className="flex flex-wrap items-center gap-3 border-b border-border/60 bg-card px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Algorithm
          </span>
          <Select value={subMode} onValueChange={(v) => setSubMode(v as RetrievalMode)}>
            <SelectTrigger className="h-8 w-[240px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RETRIEVAL_MODES.map((m) => (
                <SelectItem key={m.id} value={m.id} className="text-xs">
                  {m.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Tooltip>
            <TooltipTrigger
              type="button"
              aria-label="Algorithm description"
              className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-transparent text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <HelpCircle className="h-3.5 w-3.5" />
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-xs">
              <div className="space-y-2 text-left">
                <p className="font-medium">{activeModeMeta.name}</p>
                <p className="text-muted-foreground">{activeModeMeta.description}</p>
                <div className="border-t border-border/40 pt-1.5 text-[10px] leading-snug text-muted-foreground">
                  All four algorithms share a Weisfeiler-Lehman feature space.{' '}
                  <span className="font-medium text-foreground">Cosine</span> compares whole graphs,{' '}
                  <span className="font-medium text-foreground">containment</span> asks &quot;is query inside candidate?&quot;,{' '}
                  <span className="font-medium text-foreground">node matching</span> aligns each query node to its best local match, and{' '}
                  <span className="font-medium text-foreground">two-stage</span> uses containment to recall, then node matching to rerank.
                </div>
              </div>
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Top-k
          </span>
          <input
            type="number"
            min={1}
            max={20}
            value={k}
            onChange={(e) => setK(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
            className="h-8 w-16 rounded-md border border-border bg-background px-2 text-xs"
          />
        </div>
        <div className="flex-1" />
        {elapsedSec !== null && !loading && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {(elapsedSec * 1000).toFixed(0)} ms - {results.length} results
          </span>
        )}
        <Button
          onClick={handleRetrieve}
          disabled={loading || !queryHasNodes}
          size="sm"
          className="h-8"
        >
          {loading ? (
            <>
              <span className="mr-2 h-3 w-3 animate-spin rounded-full border-2 border-background/30 border-t-background" />
              Retrieving...
            </>
          ) : (
            <>
              <Search className="mr-1.5 h-3.5 w-3.5" />
              Retrieve
            </>
          )}
        </Button>
      </div>
      </TooltipProvider>

      {/* Middle row: query canvas (always) + optional expanded detail on the
          right (50/50 when open). The editor on the left is never replaced. */}
      <div className="flex flex-1 overflow-hidden">
        <div
          className={cn(
            'relative flex items-center justify-center overflow-hidden bg-background p-4',
            selectedMatch ? 'w-1/2 border-r border-border/60' : 'flex-1',
          )}
        >
          {!queryHasNodes && !loading && (
            <div className="pointer-events-none absolute z-10 max-w-sm rounded-xl border border-border/60 bg-card/85 px-4 py-3 text-center text-xs text-muted-foreground shadow-sm">
              Add rooms with the toolbar above the canvas, then connect them.
              This partial graph is your query.
            </div>
          )}
          <div className="h-full w-full">
            <BubbleGraphCanvas
              graph={queryGraph}
              dataset={RETRIEVAL_DATASET}
              mode="edit"
              onChange={setQueryGraph}
            />
          </div>
        </div>
        {selectedMatch && (
          <ExpandedMatchView
            rank={selectedMatch.rank}
            item={selectedMatch.item}
            onClose={() => setSelectedMatch(null)}
          />
        )}
      </div>

      {/* Bottom: top-k results strip. Fixed height matches card content so the
          strip never grows beyond the cards (avoids dead background below). */}
      <div className="flex h-[252px] flex-none flex-col overflow-hidden border-t border-border/60 bg-card">
        <div className="flex items-baseline justify-between border-b border-border/60 px-4 py-2">
          <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Top {results.length || k} matches
          </h4>
          {error && (
            <span className="rounded-md bg-destructive/10 px-2 py-0.5 text-[11px] text-destructive">
              {error}
            </span>
          )}
        </div>
        <div className="flex-1 overflow-x-auto overflow-y-hidden">
          {loading ? (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              <div className="mr-3 h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground/70" />
              Searching {/* gretrieval corpus */} corpus...
            </div>
          ) : results.length === 0 ? (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              No results yet -- draw a query and click Retrieve.
            </div>
          ) : (
            // items-start keeps each card at its natural height. Without this,
            // flex defaults to items-stretch and the card grows to the strip
            // height, leaving whitespace below the images.
            <div className="flex h-full items-start gap-3 p-3">
              {results.map((item, i) => {
                const rank = i + 1;
                const isSelected =
                  selectedMatch?.item.idx === item.idx && selectedMatch?.rank === rank;
                return (
                  <ResultCard
                    key={`${item.idx}-${i}`}
                    rank={rank}
                    item={item}
                    isSelected={isSelected}
                    onClick={() => setSelectedMatch({ item, rank })}
                  />
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


const THUMB_PX = 170;

interface ResultCardProps {
  rank: number;
  item: RetrievedItem;
  isSelected: boolean;
  onClick: () => void;
}

function ResultCard({ rank, item, isSelected, onClick }: ResultCardProps) {
  const hasFloorplan = item.floorplan_image && item.floorplan_image.length > 0;
  const hasBubble = item.bubble_image && item.bubble_image.length > 0;
  const needsRestart = !hasFloorplan && !hasBubble;
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      className={cn(
        'flex flex-shrink-0 cursor-pointer flex-col overflow-hidden rounded-lg border bg-background shadow-sm transition-all hover:shadow-md',
        isSelected
          ? 'border-foreground ring-2 ring-foreground/30'
          : 'border-border/60 hover:border-border',
      )}
      title="Click to expand to the right of the canvas"
    >
      <div className="flex items-center justify-between gap-3 border-b border-border/60 bg-card px-2.5 py-1">
        <span className="text-[10px] font-semibold tracking-tight text-foreground">
          #{rank}
        </span>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          {item.graph.num_nodes}n - {item.graph.num_edges}e
        </span>
        <span className="rounded bg-foreground/90 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-background">
          {item.score.toFixed(2)}
        </span>
      </div>
      {needsRestart ? (
        <div
          className="flex items-center justify-center px-3 text-center text-[11px] text-muted-foreground"
          style={{ width: 2 * THUMB_PX, height: THUMB_PX }}
        >
          Backend returned no images.<br />Restart uvicorn so the latest <code>graph_retrieval.py</code> loads.
        </div>
      ) : (
        <div className="flex">
          <div
            className="flex items-center justify-center overflow-hidden border-r border-border/60 bg-white"
            style={{ width: THUMB_PX, height: THUMB_PX }}
            title="Room polygons"
          >
            <img
              src={item.floorplan_image}
              alt={`Floorplan idx ${item.idx}`}
              className="block h-full w-full object-cover"
            />
          </div>
          <div
            className="flex items-center justify-center overflow-hidden bg-white"
            style={{ width: THUMB_PX, height: THUMB_PX }}
            title="Bubble graph at room centroids"
          >
            <img
              src={item.bubble_image}
              alt={`Bubble graph idx ${item.idx}`}
              className="block h-full w-full object-cover"
            />
          </div>
        </div>
      )}
    </div>
  );
}


interface ExpandedMatchViewProps {
  rank: number;
  item: RetrievedItem;
  onClose: () => void;
}

function ExpandedMatchView({ rank, item, onClose }: ExpandedMatchViewProps) {
  // Program = room-type histogram + counts, colored with the MSD palette.
  // Always render ALL 9 room types so the section doubles as a colour legend.
  const program = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const rt of item.graph.rooms) counts[rt] = (counts[rt] || 0) + 1;
    return ROOM_TYPES_MSD
      .map((meta) => ({ ...meta, count: counts[meta.id] || 0 }))
      .sort((a, b) => b.count - a.count);
  }, [item]);

  return (
    <div className="flex w-1/2 flex-col overflow-hidden border-l border-border/60 bg-background">
      <div className="flex flex-none items-center justify-between gap-3 border-b border-border/60 bg-card px-4 py-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Match #{rank}
          </span>
          <span className="text-xs font-medium text-foreground">idx {item.idx}</span>
          <span className="rounded bg-foreground/90 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-background">
            {item.score.toFixed(3)}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close detail view"
          className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Main area: candidate bubble graph fills the panel, with floorplan +
          program floating in the corners. No scroll: everything visible at once.
          The bubble graph is the matplotlib-rendered PNG (room-colored nodes,
          typed edges, centroid positions) -- visually the same as the editor
          style on the left. We can't use BubbleGraphCanvas in view mode here
          because its force simulation triggers an infinite setState loop when
          mounted read-only (a latent bug in the canvas; the editor on the left
          works because edit mode dispatches through a reducer instead). */}
      <div className="relative flex-1 overflow-hidden bg-white">
        <img
          src={item.bubble_image}
          alt={`Bubble graph idx ${item.idx}`}
          className="absolute inset-0 h-full w-full object-contain"
        />

        {/* Floorplan PNG -- floating top-right */}
        <div className="absolute right-3 top-3 w-44 overflow-hidden rounded-lg border border-border/60 bg-white shadow-lg">
          <div className="border-b border-border/60 bg-card px-2 py-1 text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
            Floorplan
          </div>
          <img
            src={item.floorplan_image}
            alt={`Floorplan idx ${item.idx}`}
            className="block w-full"
          />
        </div>

        {/* Program list -- floating bottom-right, all 9 room types listed
            (zero-count rooms muted to also serve as a colour legend) */}
        <div className="absolute bottom-3 right-3 w-44 overflow-hidden rounded-lg border border-border/60 bg-card/95 shadow-lg backdrop-blur-sm">
          <div className="flex items-baseline justify-between border-b border-border/60 bg-card px-2 py-1">
            <span className="text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
              Program
            </span>
            <span className="text-[9px] tabular-nums text-muted-foreground">
              {item.graph.num_nodes}r · {item.graph.num_edges}c
            </span>
          </div>
          <div className="divide-y divide-border/30">
            {program.map((p) => (
              <div
                key={p.id}
                className={cn(
                  'flex items-center gap-2 px-2 py-0.5 transition-opacity',
                  p.count === 0 && 'opacity-40',
                )}
              >
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm border border-black/10"
                  style={{ background: p.color }}
                />
                <span className="flex-1 truncate text-[11px] text-foreground">{p.name}</span>
                <span className="text-[11px] tabular-nums text-muted-foreground">
                  ×{p.count}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
