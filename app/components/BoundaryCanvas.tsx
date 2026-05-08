'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { PenTool, RotateCcw, Lock, Unlock, Square, Minus, Plus, Undo2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Props {
  onGenerate: (boundaryImage: string) => void;
  loading: boolean;
  value?: string;
  onChange?: (dataUrl: string) => void;
}

type DrawMode = 'polygon' | 'rect-add' | 'rect-sub';

interface Rect {
  x: number; y: number; w: number; h: number;
  op: 'add' | 'sub';
}

const CANVAS_DISPLAY_SIZE = 400;
const EXPORT_SIZE = 256;
const CLOSE_THRESHOLD = 12;

export function BoundaryCanvas({ onGenerate, loading, onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Polygon mode state
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [closed, setClosed] = useState(false);
  const [hoveredClose, setHoveredClose] = useState(false);
  const [snapAxis, setSnapAxis] = useState(true);
  const [previewPoint, setPreviewPoint] = useState<{ x: number; y: number } | null>(null);

  // Rectangle mode state
  const [rects, setRects] = useState<Rect[]>([]);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragEnd, setDragEnd] = useState<{ x: number; y: number } | null>(null);

  const [drawMode, setDrawMode] = useState<DrawMode>('rect-add');

  const isPolygon = drawMode === 'polygon';
  const isRect = drawMode === 'rect-add' || drawMode === 'rect-sub';
  const hasContent = isPolygon ? points.length > 0 : rects.length > 0;
  const canGenerate = isPolygon ? closed : rects.length > 0;

  const snapToAxis = useCallback(
    (pt: { x: number; y: number }): { x: number; y: number } => {
      if (!snapAxis || points.length === 0) return pt;
      const last = points[points.length - 1];
      const dx = Math.abs(pt.x - last.x);
      const dy = Math.abs(pt.y - last.y);
      return dx < dy ? { x: last.x, y: pt.y } : { x: pt.x, y: last.y };
    },
    [snapAxis, points]
  );

  // ── Rasterize the boolean rect union onto a temp canvas ──
  const rasterizeRects = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, rectList: Rect[]) => {
    ctx.clearRect(0, 0, w, h);
    for (const r of rectList) {
      if (r.op === 'add') {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'black';
        ctx.fillRect(r.x, r.y, r.w, r.h);
      } else {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'black';
        ctx.fillRect(r.x, r.y, r.w, r.h);
      }
    }
    ctx.globalCompositeOperation = 'source-over';
  }, []);

  // ── Draw the display canvas ──
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    if (displayW === 0 || displayH === 0) return;
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, displayW, displayH);

    // Grid
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.04)';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x < displayW; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, displayH); ctx.stroke();
    }
    for (let y = 0; y < displayH; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(displayW, y); ctx.stroke();
    }

    if (isRect) {
      // Build the boolean shape on an offscreen canvas
      const allRects = [...rects];
      if (dragStart && dragEnd) {
        allRects.push({
          x: Math.min(dragStart.x, dragEnd.x),
          y: Math.min(dragStart.y, dragEnd.y),
          w: Math.abs(dragEnd.x - dragStart.x),
          h: Math.abs(dragEnd.y - dragStart.y),
          op: drawMode === 'rect-add' ? 'add' : 'sub',
        });
      }

      if (allRects.length > 0) {
        // Rasterize boolean result
        const off = document.createElement('canvas');
        off.width = displayW; off.height = displayH;
        const offCtx = off.getContext('2d')!;
        rasterizeRects(offCtx, displayW, displayH, allRects);

        // Get merged shape pixels for edge detection
        const imgData = offCtx.getImageData(0, 0, displayW, displayH);
        const px = imgData.data;
        const isFilled = (x: number, y: number) => {
          if (x < 0 || x >= displayW || y < 0 || y >= displayH) return false;
          return px[(y * displayW + x) * 4 + 3] > 0;
        };

        // Draw filled merged shape
        ctx.save();
        ctx.globalAlpha = 0.08;
        ctx.drawImage(off, 0, 0);
        ctx.globalAlpha = 1;
        ctx.restore();

        // Draw merged outline by finding edge pixels
        ctx.fillStyle = 'oklch(0.205 0 0)';
        for (let y = 0; y < displayH; y++) {
          for (let x = 0; x < displayW; x++) {
            if (isFilled(x, y)) {
              if (!isFilled(x - 1, y) || !isFilled(x + 1, y) || !isFilled(x, y - 1) || !isFilled(x, y + 1)) {
                ctx.fillRect(x, y, 1.5, 1.5);
              }
            }
          }
        }

        // Draw drag preview rectangle outline (dashed)
        if (dragStart && dragEnd) {
          const x = Math.min(dragStart.x, dragEnd.x);
          const y = Math.min(dragStart.y, dragEnd.y);
          const w = Math.abs(dragEnd.x - dragStart.x);
          const h = Math.abs(dragEnd.y - dragStart.y);
          ctx.strokeStyle = drawMode === 'rect-add' ? 'oklch(0.45 0.2 145)' : 'oklch(0.577 0.245 27.325)';
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(x, y, w, h);
          ctx.setLineDash([]);
        }
      }
    } else {
      // Polygon mode (unchanged logic)
      if (points.length > 0) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
        if (previewPoint && !closed) ctx.lineTo(previewPoint.x, previewPoint.y);
        if (closed) {
          ctx.closePath();
          ctx.fillStyle = 'rgba(0, 0, 0, 0.04)';
          ctx.fill();
        }
        ctx.strokeStyle = 'oklch(0.205 0 0)';
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.stroke();
      }

      if (previewPoint && !closed && points.length > 0) {
        const last = points[points.length - 1];
        ctx.beginPath(); ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'oklch(0.5 0 0 / 0.3)'; ctx.lineWidth = 1;
        ctx.moveTo(last.x, last.y); ctx.lineTo(previewPoint.x, previewPoint.y);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(previewPoint.x, previewPoint.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'oklch(0.5 0 0 / 0.4)'; ctx.fill();
      }

      points.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, i === 0 && hoveredClose ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0
          ? hoveredClose ? 'oklch(0.577 0.245 27.325)' : 'oklch(0.205 0 0)'
          : 'oklch(0.439 0 0)';
        ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
      });
    }
  }, [points, closed, hoveredClose, previewPoint, rects, dragStart, dragEnd, drawMode, isRect, rasterizeRects]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);
  useEffect(() => {
    const observer = new ResizeObserver(() => drawCanvas());
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [drawCanvas]);

  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  // ── Polygon handlers ──
  const handlePolygonClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (closed) return;
      const raw = getCanvasPoint(e);
      if (!raw) return;
      const pt = snapToAxis(raw);
      if (points.length >= 3) {
        const dx = pt.x - points[0].x;
        const dy = pt.y - points[0].y;
        if (Math.sqrt(dx * dx + dy * dy) < CLOSE_THRESHOLD) {
          setClosed(true); setPreviewPoint(null); return;
        }
      }
      setPoints((prev) => [...prev, pt]);
    }, [points, closed, snapToAxis]
  );

  const handlePolygonMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (closed) { setHoveredClose(false); setPreviewPoint(null); return; }
      const raw = getCanvasPoint(e);
      if (!raw) return;
      const pt = snapToAxis(raw);
      setPreviewPoint(pt);
      if (points.length >= 3) {
        const dx = pt.x - points[0].x; const dy = pt.y - points[0].y;
        setHoveredClose(Math.sqrt(dx * dx + dy * dy) < CLOSE_THRESHOLD);
      } else { setHoveredClose(false); }
    }, [points, closed, snapToAxis]
  );

  // ── Rectangle handlers ──
  const handleRectMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = getCanvasPoint(e);
    if (pt) setDragStart(pt);
  }, []);

  const handleRectMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragStart) return;
    const pt = getCanvasPoint(e);
    if (pt) setDragEnd(pt);
  }, [dragStart]);

  const handleRectMouseUp = useCallback(() => {
    if (dragStart && dragEnd) {
      const w = Math.abs(dragEnd.x - dragStart.x);
      const h = Math.abs(dragEnd.y - dragStart.y);
      if (w > 5 && h > 5) {
        const newRect: Rect = {
          x: Math.min(dragStart.x, dragEnd.x),
          y: Math.min(dragStart.y, dragEnd.y),
          w, h,
          op: drawMode === 'rect-add' ? 'add' : 'sub',
        };
        setRects(prev => [...prev, newRect]);
      }
    }
    setDragStart(null);
    setDragEnd(null);
  }, [dragStart, dragEnd, drawMode]);

  const undoRect = useCallback(() => {
    setRects(prev => prev.slice(0, -1));
  }, []);

  // ── Clear ──
  const clearAll = useCallback(() => {
    setPoints([]); setClosed(false); setHoveredClose(false); setPreviewPoint(null);
    setRects([]); setDragStart(null); setDragEnd(null);
  }, []);

  // ── Export ──
  const exportBoundary = useCallback((): string => {
    const offscreen = document.createElement('canvas');
    offscreen.width = EXPORT_SIZE;
    offscreen.height = EXPORT_SIZE;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return '';

    // White background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, EXPORT_SIZE, EXPORT_SIZE);

    const canvas = canvasRef.current;
    const displayW = canvas?.clientWidth || CANVAS_DISPLAY_SIZE;
    const displayH = canvas?.clientHeight || CANVAS_DISPLAY_SIZE;
    const scaleX = EXPORT_SIZE / displayW;
    const scaleY = EXPORT_SIZE / displayH;

    if (isRect && rects.length > 0) {
      // Rasterize boolean rects to a mask, then draw outline on export canvas
      const mask = document.createElement('canvas');
      mask.width = EXPORT_SIZE; mask.height = EXPORT_SIZE;
      const mCtx = mask.getContext('2d')!;

      // Build the boolean shape
      for (const r of rects) {
        const rx = r.x * scaleX, ry = r.y * scaleY, rw = r.w * scaleX, rh = r.h * scaleY;
        if (r.op === 'add') {
          mCtx.globalCompositeOperation = 'source-over';
          mCtx.fillStyle = 'white';
          mCtx.fillRect(rx, ry, rw, rh);
        } else {
          mCtx.globalCompositeOperation = 'destination-out';
          mCtx.fillStyle = 'white';
          mCtx.fillRect(rx, ry, rw, rh);
        }
      }
      mCtx.globalCompositeOperation = 'source-over';

      // Extract the boundary outline from the mask using edge detection
      const maskData = mCtx.getImageData(0, 0, EXPORT_SIZE, EXPORT_SIZE);
      const pixels = maskData.data;
      const isWhite = (x: number, y: number) => {
        if (x < 0 || x >= EXPORT_SIZE || y < 0 || y >= EXPORT_SIZE) return false;
        return pixels[(y * EXPORT_SIZE + x) * 4 + 3] > 0;
      };

      // Find edge pixels (white pixel with at least one non-white neighbor)
      const edgePixels: [number, number][] = [];
      for (let y = 0; y < EXPORT_SIZE; y++) {
        for (let x = 0; x < EXPORT_SIZE; x++) {
          if (isWhite(x, y)) {
            if (!isWhite(x-1, y) || !isWhite(x+1, y) || !isWhite(x, y-1) || !isWhite(x, y+1)) {
              edgePixels.push([x, y]);
            }
          }
        }
      }

      // Draw thick black lines on edge pixels (matching training format)
      for (const [x, y] of edgePixels) {
        // 7x7 black square centered on each edge pixel
        ctx.fillStyle = 'black';
        ctx.fillRect(x - 3, y - 3, 7, 7);
      }

    } else if (!isRect && points.length >= 3) {
      // Polygon mode
      const scaled = points.map(p => ({ x: p.x * scaleX, y: p.y * scaleY }));
      for (const width of [7, 5, 3, 1]) {
        ctx.beginPath();
        ctx.moveTo(scaled[0].x, scaled[0].y);
        for (let i = 1; i < scaled.length; i++) ctx.lineTo(scaled[i].x, scaled[i].y);
        ctx.closePath();
        ctx.strokeStyle = 'black'; ctx.lineWidth = width; ctx.lineJoin = 'round';
        ctx.stroke();
      }
      for (const p of scaled) {
        ctx.fillStyle = 'black';
        ctx.fillRect(p.x - 3, p.y - 3, 7, 7);
      }
    }

    return offscreen.toDataURL('image/png');
  }, [points, rects, isRect]);

  const handleGenerate = useCallback(() => {
    const dataUrl = exportBoundary();
    const base64 = dataUrl.split(',')[1] || '';
    onGenerate(base64);
  }, [exportBoundary, onGenerate]);

  // Notify parent of draft changes after each stroke/clear
  useEffect(() => {
    if (!onChange) return;
    // Only emit when there is actual content
    if (!canGenerate) {
      onChange('');
      return;
    }
    const dataUrl = exportBoundary();
    onChange(dataUrl);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [closed, rects, points.length]);

  // ── Event dispatcher ──
  const onCanvasClick = isPolygon ? handlePolygonClick : undefined;
  const onCanvasMouseDown = isRect ? handleRectMouseDown : undefined;
  const onCanvasMouseMove = isRect ? handleRectMouseMove : handlePolygonMove;
  const onCanvasMouseUp = isRect ? handleRectMouseUp : undefined;

  return (
    <div className="flex flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-1">
          {/* Mode buttons */}
          <div className="flex rounded-lg border border-border/80 overflow-hidden">
            <button
              onClick={() => setDrawMode('rect-add')}
              className={`flex items-center gap-1 px-2 py-1.5 text-[11px] font-medium transition-colors ${
                drawMode === 'rect-add' ? 'bg-foreground text-background' : 'hover:bg-muted'
              }`}
              title="Add rectangle"
            >
              <Plus className="h-3 w-3" /><Square className="h-3 w-3" />
            </button>
            <button
              onClick={() => setDrawMode('rect-sub')}
              className={`flex items-center gap-1 px-2 py-1.5 text-[11px] font-medium border-l border-border/80 transition-colors ${
                drawMode === 'rect-sub' ? 'bg-destructive text-destructive-foreground' : 'hover:bg-muted'
              }`}
              title="Subtract rectangle"
            >
              <Minus className="h-3 w-3" /><Square className="h-3 w-3" />
            </button>
            <button
              onClick={() => setDrawMode('polygon')}
              className={`flex items-center gap-1 px-2 py-1.5 text-[11px] font-medium border-l border-border/80 transition-colors ${
                drawMode === 'polygon' ? 'bg-foreground text-background' : 'hover:bg-muted'
              }`}
              title="Freeform polygon"
            >
              <PenTool className="h-3 w-3" />
            </button>
          </div>

          {isPolygon && (
            <Button
              variant={snapAxis ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setSnapAxis(!snapAxis)}
              className="text-xs h-7 px-2"
            >
              {snapAxis ? <Lock className="mr-1 h-3 w-3" /> : <Unlock className="mr-1 h-3 w-3" />}
              {snapAxis ? 'Snap' : 'Free'}
            </Button>
          )}
        </div>

        <div className="flex items-center gap-1">
          {isRect && rects.length > 0 && (
            <Button variant="ghost" size="sm" onClick={undoRect} className="text-xs h-7 px-2">
              <Undo2 className="mr-1 h-3 w-3" />Undo
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={clearAll} disabled={!hasContent} className="text-xs h-7 px-2">
            <RotateCcw className="mr-1 h-3 w-3" />Clear
          </Button>
        </div>
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border/40 px-4 py-1.5 bg-muted/20">
        <span className="text-[11px] text-muted-foreground">
          {isRect
            ? drawMode === 'rect-add'
              ? `Draw rectangles to add (+). ${rects.length} rect${rects.length !== 1 ? 's' : ''}`
              : `Draw rectangles to subtract (−). ${rects.length} rect${rects.length !== 1 ? 's' : ''}`
            : closed
              ? 'Polygon closed'
              : `Click to place vertices. ${points.length} point${points.length !== 1 ? 's' : ''}`
          }
        </span>
      </div>

      {/* Canvas area */}
      <div className="relative flex-1 flex items-center justify-center bg-muted/30 p-4 min-h-[300px]">
        {!hasContent && !(dragStart && dragEnd) && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none">
            {isRect ? (
              <>
                <Square className="h-8 w-8 text-muted-foreground/30 mb-2" />
                <p className="text-sm text-muted-foreground/60">Drag to draw rectangles</p>
                <p className="text-xs text-muted-foreground/40 mt-1">Use +□ to add, −□ to subtract</p>
              </>
            ) : (
              <>
                <PenTool className="h-8 w-8 text-muted-foreground/30 mb-2" />
                <p className="text-sm text-muted-foreground/60">Click to draw a boundary polygon</p>
                <p className="text-xs text-muted-foreground/40 mt-1">
                  {snapAxis ? 'Snap mode: lines auto-align to H/V' : 'Click near the first point to close'}
                </p>
              </>
            )}
          </div>
        )}
        <canvas
          ref={canvasRef}
          onClick={onCanvasClick}
          onMouseDown={onCanvasMouseDown}
          onMouseMove={onCanvasMouseMove}
          onMouseUp={onCanvasMouseUp}
          onMouseLeave={isRect ? handleRectMouseUp : undefined}
          className={`w-full h-full rounded-xl border border-border/60 shadow-sm ${
            isRect ? 'cursor-crosshair' : closed ? 'cursor-default' : 'cursor-crosshair'
          }`}
          style={{ maxWidth: CANVAS_DISPLAY_SIZE, maxHeight: CANVAS_DISPLAY_SIZE, aspectRatio: '1 / 1' }}
        />
      </div>

      {/* Generate */}
      <div className="border-t border-border/60 p-4">
        {isPolygon && !closed && points.length > 0 && (
          <p className="mb-2 text-center text-xs text-muted-foreground">
            {points.length < 3
              ? `Place ${3 - points.length} more point${3 - points.length > 1 ? 's' : ''}, then close`
              : 'Click near the first point to close the polygon'}
          </p>
        )}
        <Button
          onClick={handleGenerate}
          disabled={!canGenerate || loading}
          className="w-full h-10 text-sm font-medium"
          size="lg"
        >
          {loading ? (
            <>
              <div className="mr-2 h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
              Generating...
            </>
          ) : (
            <>
              <PenTool className="mr-2 h-3.5 w-3.5" />
              Generate from Boundary
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
