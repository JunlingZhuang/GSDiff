"use client";

import * as React from "react";
import { Check, Lock, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ROOM_PRESETS, type RoomStudio } from "@/hooks/use-room-studio";
import { ICU_CATALOG } from "@/lib/icu-catalog";
import { cn } from "@/lib/utils";

// The clear-floor-area floor from the ICU schematic rules (icu_room_rules.json:
// room.min_clear_area_sf). Below it the live readout turns warning-toned.
const MIN_CLEAR_AREA_SF = 200;

// Parse a dimension field to a number, tolerating an in-progress empty/partial
// entry (kept as the raw string in local state so the caret does not jump).
function DimensionInput({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: number;
  onCommit: (next: number) => void;
}) {
  const [text, setText] = React.useState(String(value));
  // Re-sync when the committed value changes from outside this field.
  React.useEffect(() => setText(String(value)), [value]);

  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    if (Number.isFinite(parsed)) onCommit(parsed);
    else setText(String(value));
  };

  return (
    <div className="flex items-center gap-1.5">
      <Label className="w-9 text-[11px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        inputMode="decimal"
        min={8}
        max={40}
        step={0.25}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        className="h-8 font-mono text-[13px] tabular-nums"
      />
    </div>
  );
}

// v1 ships every catalog item as a required, locked selection (the checkbox is
// reserved for a later editable version). Renders the drafting symbol, label,
// footprint, and required count.
function AssetChecklist() {
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      {ICU_CATALOG.map((entry, index) => (
        <div
          key={entry.type}
          className={cn(
            "flex items-center gap-2.5 px-2 py-1.5",
            index > 0 && "border-t border-border/60",
          )}
        >
          <span
            className="flex size-4 shrink-0 items-center justify-center rounded-[4px] bg-primary text-primary-foreground"
            title="Required in ICU Room v1 (locked)"
          >
            <Check className="size-2.5" />
          </span>
          <span className="flex size-7 shrink-0 items-center justify-center rounded-md border border-border bg-secondary/30">
            <img src={entry.symbol} alt="" className="max-h-5 max-w-5 object-contain" />
          </span>
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-[12px] text-foreground">{entry.label}</span>
            <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
              {entry.footprint_ft[0]} × {entry.footprint_ft[1]} ft · {entry.anchor}
            </span>
          </div>
          {entry.count_required > 1 ? (
            <span className="shrink-0 rounded-md border border-border bg-secondary/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-muted-foreground">
              ×{entry.count_required}
            </span>
          ) : null}
          <Lock className="size-3 shrink-0 text-muted-foreground/60" aria-hidden />
        </div>
      ))}
    </div>
  );
}

export function RoomPanel({ room }: { room: RoomStudio }) {
  const area = Math.round(room.areaFt2);
  const areaWarn = room.areaFt2 < MIN_CLEAR_AREA_SF;
  const requiredCount = ICU_CATALOG.reduce((total, entry) => total + entry.count_required, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* HEADER — fixed: the preset picker names the layout program. */}
      <div className="flex shrink-0 flex-col gap-1.5 border-b border-border px-3 py-2.5">
        <span className="label-xs">Preset</span>
        <Select value={room.preset} onValueChange={(value) => value && room.setPreset(value)}>
          <SelectTrigger className="h-8 w-full text-[13px]">
            <SelectValue placeholder="Pick a preset" />
          </SelectTrigger>
          <SelectContent>
            {ROOM_PRESETS.map((option) => (
              <SelectItem key={option.key} value={option.key} className="text-[13px]">
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* BODY — the single scroll context for the room program. */}
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 p-3">
          {/* dimensions */}
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="label-xs">Dimensions</span>
              <span
                className={cn(
                  "font-mono text-[12px] tabular-nums",
                  areaWarn ? "text-warning" : "text-muted-foreground",
                )}
              >
                {area} ft²
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <DimensionInput label="Width" value={room.widthFt} onCommit={room.setWidthFt} />
              <DimensionInput label="Depth" value={room.depthFt} onCommit={room.setDepthFt} />
            </div>
            {areaWarn ? (
              <p className="text-[11px] text-warning">
                Below the {MIN_CLEAR_AREA_SF} sf clear-area planning minimum.
              </p>
            ) : (
              <p className="text-[11px] text-muted-foreground">
                Feet, snapped to 0.25 ft. West/South is the origin corner.
              </p>
            )}
          </section>

          {/* asset checklist */}
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="label-xs">Assets</span>
              <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                {requiredCount} required
              </span>
            </div>
            <AssetChecklist />
            <p className="text-[11px] text-muted-foreground">
              Every item is required and locked in v1; the validator places them.
            </p>
          </section>

          {/* brief */}
          <section className="flex flex-col gap-2">
            <span className="label-xs">Brief</span>
            <Textarea
              value={room.prompt}
              onChange={(event) => room.setPrompt(event.target.value)}
              placeholder="Calm room, equipment on the east side, visitor chair by the window."
              className="min-h-20 max-h-40 resize-y text-[12px] leading-relaxed"
            />
          </section>
        </div>
      </ScrollArea>

      {/* FOOTER — fixed: primary action + status line. */}
      <div className="shrink-0 border-t border-border p-3">
        <Button
          className="h-9 w-full gap-2 text-[13px] font-medium"
          disabled={!!room.busyAction}
          onClick={() => void room.generateRoom()}
        >
          <Sparkles className="size-3.5" />
          {room.busyAction === "generate"
            ? "Generating…"
            : room.result
              ? "Re-generate room"
              : "Generate room"}
        </Button>
        <p
          className={cn(
            "mt-2 truncate font-mono text-[11px] tabular-nums",
            room.status.error ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {room.status.text}
        </p>
      </div>
    </div>
  );
}
