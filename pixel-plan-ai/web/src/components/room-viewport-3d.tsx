"use client";

import * as React from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls, useGLTF } from "@react-three/drei";
import type { Group } from "three";

import { ICU_CATALOG, ICU_CATALOG_BY_TYPE } from "@/lib/icu-catalog";
import type { RoomLayout, RoomWall } from "@/lib/types";

// The scene works in FEET (1 world unit = 1 ft) — the room contract's native
// unit — so all shell geometry reads directly against the backend numbers. The
// GLB library is authored in meters (1 ft = 0.3048 m per the generator spec),
// so every loaded instance is scaled by GLB_TO_FT at the <primitive> boundary.
// One unit adapter, in one place.
const GLB_TO_FT = 1 / 0.3048; // = 3.28084

// Dollhouse conventions, mirroring the floor flow's plan3d.ts: waist-height
// light walls (plan3d WALL_HEIGHT = 1.5 m ≈ 4.95 ft, i.e. ~55% of the 9 ft
// nominal ceiling) keep the interior visible from an orbiting camera, and the
// door head sits at 85% of the wall height with a lintel band above the
// opening. There is NO solid ceiling: the 9 ft ceiling plane is indicated only
// by the boom drops that hang from it.
const CEILING_FT = 9;
const WALL_H_FT = 0.55 * CEILING_FT;
const DOOR_H_FT = WALL_H_FT * 0.85;
const WALL_T_FT = 0.5;
const FLOOR_T_FT = 0.33;
const WALL_COLOR = "#f1efe9";
const FLOOR_COLOR = "#ece9e2";

// Preload every catalog model so the first switch to 3D does not stall per asset.
for (const entry of ICU_CATALOG) useGLTF.preload(entry.model);

interface Box {
  position: [number, number, number];
  size: [number, number, number];
}

// Build the room shell: floor slab + four waist-height walls, the door wall
// split around the opening with a lintel header above it. Feet in, feet out,
// centered on origin: room x -> world X, room y (north) -> world +Z, floor top
// at world y = 0. Walls are grouped by their room side so the viewer can hide
// the two camera-facing sides (dollhouse cutaway).
function buildShell(layout: RoomLayout): { floor: Box; walls: Record<RoomWall, Box[]> } {
  const width = layout.room.width_ft;
  const depth = layout.room.depth_ft;
  const door = layout.room.door;
  const offsetX = width / 2;
  const offsetZ = depth / 2;

  const floor: Box = { position: [0, -FLOOR_T_FT / 2, 0], size: [width, FLOOR_T_FT, depth] };
  const walls: Record<RoomWall, Box[]> = { N: [], S: [], E: [], W: [] };

  const yMid = WALL_H_FT / 2;
  const lintelY = (DOOR_H_FT + WALL_H_FT) / 2;
  const lintelH = WALL_H_FT - DOOR_H_FT;

  // Horizontal walls (S at y=0, N at y=depth) run along X; vertical walls
  // (W at x=0, E at x=width) run along Z. Each spans an extra WALL_T past the
  // corners so the four walls close.
  const addHorizontal = (side: RoomWall, yf: number) => {
    const z = yf - offsetZ;
    if (door.wall !== side) {
      walls[side].push({
        position: [0, yMid, z],
        size: [width + 2 * WALL_T_FT, WALL_H_FT, WALL_T_FT],
      });
      return;
    }
    const o0 = door.offset_ft - door.width_ft / 2;
    const o1 = door.offset_ft + door.width_ft / 2;
    const seg = (a: number, b: number, y: number, h: number) => {
      const length = b - a;
      if (length <= 1e-4) return;
      walls[side].push({ position: [(a + b) / 2 - offsetX, y, z], size: [length, h, WALL_T_FT] });
    };
    seg(-WALL_T_FT, o0, yMid, WALL_H_FT);
    seg(o1, width + WALL_T_FT, yMid, WALL_H_FT);
    seg(o0, o1, lintelY, lintelH); // lintel over the opening
  };

  const addVertical = (side: RoomWall, xf: number) => {
    const x = xf - offsetX;
    if (door.wall !== side) {
      walls[side].push({
        position: [x, yMid, 0],
        size: [WALL_T_FT, WALL_H_FT, depth + 2 * WALL_T_FT],
      });
      return;
    }
    const o0 = door.offset_ft - door.width_ft / 2;
    const o1 = door.offset_ft + door.width_ft / 2;
    const seg = (a: number, b: number, y: number, h: number) => {
      const length = b - a;
      if (length <= 1e-4) return;
      walls[side].push({ position: [x, y, (a + b) / 2 - offsetZ], size: [WALL_T_FT, h, length] });
    };
    seg(-WALL_T_FT, o0, yMid, WALL_H_FT);
    seg(o1, depth + WALL_T_FT, yMid, WALL_H_FT);
    seg(o0, o1, lintelY, lintelH);
  };

  addHorizontal("S", 0);
  addHorizontal("N", depth);
  addVertical("W", 0);
  addVertical("E", width);

  return { floor, walls };
}

// Outward normal (world XZ) per room side, for the camera-facing test.
const SIDE_NORMAL: Record<RoomWall, [number, number]> = {
  N: [0, 1],
  S: [0, -1],
  E: [1, 0],
  W: [-1, 0],
};

// The four wall-side groups, with the two sides facing the camera hidden every
// frame (classic dollhouse cutaway) so the interior and every asset stay
// visible from any orbit angle — including wall-flush casework/sink that a
// near waist wall would otherwise occlude.
function DollhouseWalls({ walls }: { walls: Record<RoomWall, Box[]> }) {
  const refs = React.useRef<Partial<Record<RoomWall, Group | null>>>({});

  useFrame(({ camera }) => {
    for (const side of ["N", "S", "E", "W"] as RoomWall[]) {
      const group = refs.current[side];
      if (!group) continue;
      const [nx, nz] = SIDE_NORMAL[side];
      group.visible = camera.position.x * nx + camera.position.z * nz <= 0;
    }
  });

  return (
    <>
      {(["N", "S", "E", "W"] as RoomWall[]).map((side) => (
        <group
          key={side}
          ref={(node) => {
            refs.current[side] = node;
          }}
        >
          {walls[side].map((wall, index) => (
            <mesh key={index} position={wall.position}>
              <boxGeometry args={wall.size} />
              <meshStandardMaterial color={WALL_COLOR} roughness={0.9} />
            </mesh>
          ))}
        </group>
      ))}
    </>
  );
}

interface Placement {
  id: string;
  url: string;
  position: [number, number, number];
  rotationY: number;
}

function buildPlacements(layout: RoomLayout): Placement[] {
  const offsetX = layout.room.width_ft / 2;
  const offsetZ = layout.room.depth_ft / 2;

  const placements: Placement[] = [];
  for (const asset of layout.assets) {
    const catalog = ICU_CATALOG_BY_TYPE[asset.type];
    if (!catalog) continue;
    const centerX = asset.x_ft + asset.w_ft / 2 - offsetX;
    const centerZ = asset.y_ft + asset.d_ft / 2 - offsetZ;
    // Ceiling assets hang from the ceiling plane (their GLB origin is the mount
    // plate); the rest stand on the floor. Room rotation is CCW in a y-up
    // frame, which is a negative rotation about world Y (three maps local +X
    // to (cos, 0, -sin)).
    const y = asset.anchor === "ceiling" ? CEILING_FT : 0;
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
  // Meters-authored GLB → feet-unit scene, adapted here and nowhere else.
  return <primitive object={object} position={position} rotation={[0, rotationY, 0]} scale={GLB_TO_FT} />;
}

export function RoomViewport3D({ layout }: { layout: RoomLayout }) {
  const shell = React.useMemo(() => buildShell(layout), [layout]);
  const placements = React.useMemo(() => buildPlacements(layout), [layout]);
  const extent = Math.max(layout.room.width_ft, layout.room.depth_ft);
  const distance = Math.max(20, extent * 1.55);

  return (
    <Canvas
      camera={{ position: [distance * 0.72, distance * 0.62, distance * 0.72], fov: 42, near: 0.1, far: distance * 12 }}
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

      <DollhouseWalls walls={shell.walls} />

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
        position={[0, -FLOOR_T_FT - 0.05, 0]}
        args={[extent * 6, extent * 6]}
        cellSize={1}
        cellColor="#d6d3ca"
        sectionSize={4}
        sectionColor="#c2beb2"
        fadeDistance={distance * 3.2}
        infiniteGrid
      />
      <OrbitControls makeDefault maxPolarAngle={Math.PI / 2.05} minDistance={5} maxDistance={distance * 4} />
    </Canvas>
  );
}
