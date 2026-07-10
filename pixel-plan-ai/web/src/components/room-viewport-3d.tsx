"use client";

import * as React from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, useGLTF } from "@react-three/drei";

import { ICU_CATALOG, ICU_CATALOG_BY_TYPE } from "@/lib/icu-catalog";
import type { RoomLayout } from "@/lib/types";

const FT = 0.3048;
const WALL_H_FT = 9;
const WALL_T_FT = 0.5;
const DOOR_H_FT = 7;
const WALL_COLOR = "#f1efe9";
const FLOOR_COLOR = "#ece9e2";

// Preload every catalog model so the first switch to 3D does not stall per asset.
for (const entry of ICU_CATALOG) useGLTF.preload(entry.model);

interface Box {
  position: [number, number, number];
  size: [number, number, number];
}

// Build the room shell (floor slab + four walls, the door wall split around the
// opening with a lintel header above it). Feet in, meters out, centered on
// origin: room x -> world X, room y (north) -> world Z, floor top at y = 0.
function buildShell(layout: RoomLayout): { floor: Box; walls: Box[] } {
  const width = layout.room.width_ft;
  const depth = layout.room.depth_ft;
  const door = layout.room.door;
  const offsetX = (width * FT) / 2;
  const offsetZ = (depth * FT) / 2;
  const worldX = (xf: number) => xf * FT - offsetX;
  const worldZ = (yf: number) => yf * FT - offsetZ;

  const floor: Box = { position: [0, -0.05, 0], size: [width * FT, 0.1, depth * FT] };
  const walls: Box[] = [];

  const yMid = (WALL_H_FT * FT) / 2;
  const lintelY = ((DOOR_H_FT + WALL_H_FT) * FT) / 2;
  const lintelH = (WALL_H_FT - DOOR_H_FT) * FT;

  // Horizontal walls (S at y=0, N at y=depth) run along X; vertical walls
  // (W at x=0, E at x=width) run along Z. Each spans an extra WALL_T past the
  // corners so the four walls close.
  const addHorizontal = (yf: number) => {
    const z = worldZ(yf);
    const hasDoor = door.wall === (yf === 0 ? "S" : "N");
    if (!hasDoor) {
      walls.push({
        position: [0, yMid, z],
        size: [(width + 2 * WALL_T_FT) * FT, WALL_H_FT * FT, WALL_T_FT * FT],
      });
      return;
    }
    const o0 = door.offset_ft - door.width_ft / 2;
    const o1 = door.offset_ft + door.width_ft / 2;
    const seg = (a: number, b: number, y: number, h: number) => {
      const length = (b - a) * FT;
      if (length <= 1e-4) return;
      walls.push({
        position: [((a + b) / 2) * FT - offsetX, y, z],
        size: [length, h, WALL_T_FT * FT],
      });
    };
    seg(-WALL_T_FT, o0, yMid, WALL_H_FT * FT);
    seg(o1, width + WALL_T_FT, yMid, WALL_H_FT * FT);
    seg(o0, o1, lintelY, lintelH); // lintel over the opening
  };

  const addVertical = (xf: number) => {
    const x = worldX(xf);
    const hasDoor = door.wall === (xf === 0 ? "W" : "E");
    if (!hasDoor) {
      walls.push({
        position: [x, yMid, 0],
        size: [WALL_T_FT * FT, WALL_H_FT * FT, (depth + 2 * WALL_T_FT) * FT],
      });
      return;
    }
    const o0 = door.offset_ft - door.width_ft / 2;
    const o1 = door.offset_ft + door.width_ft / 2;
    const seg = (a: number, b: number, y: number, h: number) => {
      const length = (b - a) * FT;
      if (length <= 1e-4) return;
      walls.push({
        position: [x, y, ((a + b) / 2) * FT - offsetZ],
        size: [WALL_T_FT * FT, h, length],
      });
    };
    seg(-WALL_T_FT, o0, yMid, WALL_H_FT * FT);
    seg(o1, depth + WALL_T_FT, yMid, WALL_H_FT * FT);
    seg(o0, o1, lintelY, lintelH);
  };

  addHorizontal(0);
  addHorizontal(depth);
  addVertical(0);
  addVertical(width);

  return { floor, walls };
}

interface Placement {
  id: string;
  url: string;
  position: [number, number, number];
  rotationY: number;
}

function buildPlacements(layout: RoomLayout): Placement[] {
  const width = layout.room.width_ft;
  const depth = layout.room.depth_ft;
  const offsetX = (width * FT) / 2;
  const offsetZ = (depth * FT) / 2;
  const ceilingY = WALL_H_FT * FT;

  const placements: Placement[] = [];
  for (const asset of layout.assets) {
    const catalog = ICU_CATALOG_BY_TYPE[asset.type];
    if (!catalog) continue;
    const centerX = (asset.x_ft + asset.w_ft / 2) * FT - offsetX;
    const centerZ = (asset.y_ft + asset.d_ft / 2) * FT - offsetZ;
    // Ceiling assets hang from the ceiling plate (their GLB origin); the rest
    // stand on the floor. Room rotation is CCW in a y-up frame, which is a
    // negative rotation about world Y (three maps local +X to (cos, 0, -sin)).
    const y = asset.anchor === "ceiling" ? ceilingY : 0;
    placements.push({
      id: asset.id,
      url: catalog.model,
      position: [centerX, y, centerZ],
      rotationY: -(asset.rotation_deg * Math.PI) / 180,
    });
  }
  return placements;
}

function AssetModel({ url, position, rotationY }: { url: string; position: [number, number, number]; rotationY: number }) {
  const { scene } = useGLTF(url);
  // Clone so repeated types (e.g. two booms) render as independent objects.
  const object = React.useMemo(() => scene.clone(true), [scene]);
  return <primitive object={object} position={position} rotation={[0, rotationY, 0]} />;
}

export function RoomViewport3D({ layout }: { layout: RoomLayout }) {
  const shell = React.useMemo(() => buildShell(layout), [layout]);
  const placements = React.useMemo(() => buildPlacements(layout), [layout]);
  const extent = Math.max(layout.room.width_ft, layout.room.depth_ft) * FT;
  const distance = Math.max(6, extent * 1.7);

  return (
    <Canvas
      camera={{ position: [distance * 0.72, distance * 0.68, distance * 0.72], fov: 42, near: 0.1, far: distance * 12 }}
      dpr={[1, 2]}
      className="h-full w-full"
    >
      <color attach="background" args={["#f0efeb"]} />
      <hemisphereLight args={["#ffffff", "#d9d5cc", 0.9]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[extent, extent * 1.7, extent * 0.6]} intensity={1.1} />
      <directionalLight position={[-extent, extent, -extent * 0.5]} intensity={0.4} />

      <mesh position={shell.floor.position} receiveShadow>
        <boxGeometry args={shell.floor.size} />
        <meshStandardMaterial color={FLOOR_COLOR} roughness={0.95} />
      </mesh>

      {shell.walls.map((wall, index) => (
        <mesh key={`wall-${index}`} position={wall.position}>
          <boxGeometry args={wall.size} />
          <meshStandardMaterial color={WALL_COLOR} roughness={0.9} />
        </mesh>
      ))}

      <React.Suspense fallback={null}>
        {placements.map((placement) => (
          <AssetModel
            key={placement.id}
            url={placement.url}
            position={placement.position}
            rotationY={placement.rotationY}
          />
        ))}
      </React.Suspense>

      <Grid
        position={[0, -0.11, 0]}
        args={[extent * 6, extent * 6]}
        cellSize={FT}
        cellColor="#d6d3ca"
        sectionSize={4 * FT}
        sectionColor="#c2beb2"
        fadeDistance={distance * 3.2}
        infiniteGrid
      />
      <OrbitControls makeDefault maxPolarAngle={Math.PI / 2.05} minDistance={2} maxDistance={distance * 4} />
    </Canvas>
  );
}
