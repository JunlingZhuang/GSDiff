"use client";

import * as React from "react";
import { Check, Copy, Lock, Minus, Plus, Sparkles, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
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

// Compact − n + stepper for a custom program's asset count, bounded by the
// catalog min/max for that type.
function CountStepper({
  value,
  min,
  max,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (next: number) => void;
}) {
  return (
    <div className="flex shrink-0 items-center gap-1">
      <Button
        type="button"
        variant="outline"
        size="icon-xs"
        disabled={value <= min}
        onClick={() => onChange(value - 1)}
        aria-label="Decrease count"
      >
        <Minus />
      </Button>
      <span className="w-4 text-center font-mono text-[12px] tabular-nums">{value}</span>
      <Button
        type="button"
        variant="outline"
        size="icon-xs"
        disabled={value >= max}
        onClick={() => onChange(value + 1)}
        aria-label="Increase count"
      >
        <Plus />
      </Button>
    </div>
  );
}

// The catalog checklist. A PRESET renders today's locked rows. A CUSTOM program
// gives every row (except icu_bed, always locked at 1) a count stepper; a row at
// count 0 dims to read as excluded.
function AssetChecklist({
  custom,
  counts,
  onCount,
}: {
  custom: boolean;
  counts: Record<string, number>;
  onCount: (type: string, next: number) => void;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      {ICU_CATALOG.map((entry, index) => {
        const count = counts[entry.type] ?? entry.count_required;
        const included = count > 0;
        const editable = custom && entry.type !== "icu_bed";
        return (
          <div
            key={entry.type}
            className={cn(
              "flex items-center gap-2.5 px-2 py-1.5",
              index > 0 && "border-t border-border/60",
              !included && "opacity-45",
            )}
          >
            <span
              className={cn(
                "flex size-4 shrink-0 items-center justify-center rounded-[4px]",
                included ? "bg-primary text-primary-foreground" : "border border-border",
              )}
              title={included ? "Included" : "Excluded (count 0)"}
            >
              {included ? <Check className="size-2.5" /> : null}
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
            {editable ? (
              <CountStepper
                value={count}
                min={entry.min_count}
                max={entry.max_count}
                onChange={(next) => onCount(entry.type, next)}
              />
            ) : (
              <>
                {count > 1 ? (
                  <span className="shrink-0 rounded-md border border-border bg-secondary/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-muted-foreground">
                    ×{count}
                  </span>
                ) : null}
                <Lock
                  className="size-3 shrink-0 text-muted-foreground/60"
                  aria-hidden
                />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function RoomPanel({ room }: { room: RoomStudio }) {
  const area = Math.round(room.areaFt2);
  const areaWarn = room.areaFt2 < MIN_CLEAR_AREA_SF;
  const activeProgram = room.activeProgram;
  const isCustom = room.isCustomActive;

  // Focus the inline rename field once, right after a duplicate creates a program.
  const renameRef = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => {
    if (room.renameFocusId && activeProgram?.id === room.renameFocusId) {
      renameRef.current?.focus();
      renameRef.current?.select();
      room.setRenameFocusId(null);
    }
  }, [room.renameFocusId, activeProgram, room]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* HEADER — the program picker: immutable presets, then custom programs. */}
      <div className="flex shrink-0 flex-col gap-1.5 border-b border-border px-3 py-2.5">
        <span className="label-xs">Program</span>
        <div className="flex items-center gap-1.5">
          <Select
            value={room.activeProgramId}
            onValueChange={(value) => value && room.setActiveProgramId(value)}
          >
            <SelectTrigger className="h-8 flex-1 text-[13px]">
              <SelectValue placeholder="Pick a program" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Presets</SelectLabel>
                {ROOM_PRESETS.map((option) => (
                  <SelectItem key={option.key} value={option.key} className="text-[13px]">
                    {option.label}
                  </SelectItem>
                ))}
              </SelectGroup>
              {room.programs.length ? (
                <>
                  <SelectSeparator />
                  <SelectGroup>
                    <SelectLabel>Custom</SelectLabel>
                    {room.programs.map((program) => (
                      <SelectItem key={program.id} value={program.id} className="text-[13px]">
                        {program.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </>
              ) : null}
            </SelectContent>
          </Select>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="size-8 shrink-0"
            onClick={() => room.createProgram()}
            title="Duplicate to an editable custom program"
            aria-label="Duplicate program"
          >
            <Copy />
          </Button>
        </div>

        {/* Rename + delete row, custom programs only (presets are immutable). */}
        {isCustom && activeProgram ? (
          <div className="flex items-center gap-1.5">
            <Input
              ref={renameRef}
              value={activeProgram.name}
              onChange={(event) => room.renameProgram(activeProgram.id, event.target.value)}
              placeholder="Program name"
              className="h-8 text-[13px]"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
              onClick={() => room.deleteProgram(activeProgram.id)}
              title="Delete this custom program"
              aria-label="Delete program"
            >
              <Trash2 />
            </Button>
          </div>
        ) : (
          <p className="text-[11px] text-muted-foreground">
            Presets are read-only. Duplicate to edit asset counts.
          </p>
        )}
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
                {room.totalAssetCount} total
              </span>
            </div>
            <AssetChecklist custom={isCustom} counts={room.activeCounts} onCount={room.setAssetCount} />
            <p className="text-[11px] text-muted-foreground">
              {isCustom
                ? "Steppers set requested counts; 0 removes an item. The validator places them."
                : "Every item is locked in a preset; the validator places them."}
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
