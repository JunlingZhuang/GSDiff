"use client";

import * as React from "react";
import { Download, FileJson } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Studio } from "@/hooks/use-studio";

export function TopBar({ studio }: { studio: Studio }) {
  const downloadJson = React.useCallback(() => {
    if (!studio.result) return;
    const blob = new Blob([JSON.stringify(studio.result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "pixel-plan.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [studio.result]);

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b bg-background px-3">
      <div className="flex items-center gap-2">
        <span className="grid grid-cols-2 gap-px" aria-hidden>
          {["#7fb5a8", "#e3b04b", "#8d9fb8", "#1f242a"].map((color) => (
            <i key={color} className="size-1.5 rounded-[1px]" style={{ background: color }} />
          ))}
        </span>
        <span className="text-xs font-bold tracking-wide">
          PIXEL<span className="text-muted-foreground">/</span>PLAN
        </span>
        <Badge variant="secondary" className="text-[9px] font-bold tracking-wider">
          STUDIO
        </Badge>
      </div>

      <div className="min-w-0 flex-1 text-center">
        <span className="truncate text-xs font-medium text-muted-foreground">
          {studio.result?.program.building_type ?? studio.program?.building_type ?? "Untitled program"}
          {studio.activePlan
            ? ` · ${studio.activePlan.width}×${studio.activePlan.height} cells · ${(studio.activePlan.meters_per_cell * 3.28084).toFixed(2)} ft/cell`
            : ""}
        </span>
      </div>

      <div className="flex items-center gap-1.5">
        {studio.activeValidation ? (
          <Badge
            variant={studio.result?.accepted === false ? "destructive" : "default"}
            className={`font-mono text-[11px] tabular-nums ${studio.result?.accepted === false ? "" : "bg-emerald-600 hover:bg-emerald-600"}`}
          >
            {studio.activeValidation.score}
          </Badge>
        ) : null}
        <Button
          size="sm"
          variant="ghost"
          className="h-7 gap-1 px-2 text-[10px] font-bold"
          disabled={!studio.result}
          onClick={() => (window as unknown as { __pixelPlanExportPng?: () => void }).__pixelPlanExportPng?.()}
        >
          <Download className="size-3" /> PNG
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-7 gap-1 px-2 text-[10px] font-bold"
          disabled={!studio.result}
          onClick={downloadJson}
        >
          <FileJson className="size-3" /> JSON
        </Button>
        <span
          className={`ml-1 flex items-center gap-1.5 text-[10px] font-medium ${
            studio.health?.gemini_configured ? "text-emerald-700" : "text-destructive"
          }`}
        >
          <i className="size-1.5 rounded-full bg-current" />
          {studio.health ? (studio.health.gemini_configured ? studio.health.model : "No API key") : "Checking…"}
        </span>
      </div>
    </header>
  );
}
