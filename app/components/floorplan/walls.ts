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
  const fmt = (n: number) => +n.toFixed(4);
  const [first, ...rest] = pts;
  return `M ${fmt(first[0])} ${fmt(first[1])} ` + rest.map((p) => `L ${fmt(p[0])} ${fmt(p[1])}`).join(' ') + ' Z';
}

// Point at parameter t in [0,1] along the wall centreline a→b; used to place openings.
export function pointAlongWall(wall: Wall, t: number): Pt {
  return [wall.a[0] + (wall.b[0] - wall.a[0]) * t, wall.a[1] + (wall.b[1] - wall.a[1]) * t];
}

// Unit normal (perpendicular) of a wall.
export function wallNormal(wall: Wall): Pt {
  const dx = wall.b[0] - wall.a[0];
  const dy = wall.b[1] - wall.a[1];
  const len = Math.hypot(dx, dy) || 1;
  return [-dy / len, dx / len];
}

// A point at parameter t along the centreline, pushed `off` metres along the
// wall normal — used to draw the two parallel faces of a double-line wall.
export function offsetAlong(wall: Wall, t: number, off: number): Pt {
  const [nx, ny] = wallNormal(wall);
  return [
    wall.a[0] + (wall.b[0] - wall.a[0]) * t + nx * off,
    wall.a[1] + (wall.b[1] - wall.a[1]) * t + ny * off,
  ];
}
