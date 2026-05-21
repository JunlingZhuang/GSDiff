import type { Wall, Pt } from '@/lib/plan';

// A wall centreline + thickness → the 4 corners of its rectangle, so the SVG
// renders the wall as a filled double-line body rather than a hairline.
export function wallQuad(wall: Wall): Pt[] {
  const [ax, ay] = wall.a;
  const [bx, by] = wall.b;
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy) || 1;
  // unit normal
  const nx = -dy / len;
  const ny = dx / len;
  const h = wall.thickness / 2;
  return [
    [ax + nx * h, ay + ny * h],
    [bx + nx * h, by + ny * h],
    [bx - nx * h, by - ny * h],
    [ax - nx * h, ay - ny * h],
  ];
}

// SVG path "d" for a closed polygon of points.
export function polyPath(pts: Pt[]): string {
  if (pts.length === 0) return '';
  const [first, ...rest] = pts;
  return `M ${first[0]} ${first[1]} ` + rest.map((p) => `L ${p[0]} ${p[1]}`).join(' ') + ' Z';
}

// Midpoint of a wall offset by t in [0,1] from a→b; used to place openings.
export function pointAlongWall(wall: Wall, t: number): Pt {
  return [wall.a[0] + (wall.b[0] - wall.a[0]) * t, wall.a[1] + (wall.b[1] - wall.a[1]) * t];
}
