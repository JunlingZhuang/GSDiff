'use client';

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { roomColor } from '@/lib/plan';
import { type WallGraph, wallSolid, roomPoly, graphBounds } from './kernel';

const WALL_H = 2.8; // metres

interface Props {
  graph: WallGraph;
}

export function FloorPlanView3D({ graph }: Props) {
  const { wallGeom, floors, center, dist } = useMemo(() => {
    const solid = wallSolid(graph);
    const shapes: THREE.Shape[] = [];
    for (const poly of solid) {
      const outer = poly[0];
      if (!outer || outer.length < 3) continue;
      const shape = new THREE.Shape(outer.map(([x, y]) => new THREE.Vector2(x, y)));
      for (let i = 1; i < poly.length; i++) {
        shape.holes.push(new THREE.Path(poly[i].map(([x, y]) => new THREE.Vector2(x, y))));
      }
      shapes.push(shape);
    }
    const wallGeom = shapes.length
      ? new THREE.ExtrudeGeometry(shapes, { depth: WALL_H, bevelEnabled: false })
      : null;

    const floors = graph.rooms.map((r) => {
      const poly = roomPoly(graph, r);
      const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
      return { id: r.id, geom: new THREE.ShapeGeometry(shape), color: roomColor(r.type) };
    });

    const b = graphBounds(graph);
    const center: [number, number] = [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2];
    const dist = Math.max(b.maxX - b.minX, b.maxY - b.minY) * 1.4 + WALL_H * 2;
    return { wallGeom, floors, center, dist };
  }, [graph]);

  return (
    <div className="h-full w-full bg-gradient-to-b from-slate-100 to-slate-200">
      <Canvas camera={{ position: [dist * 0.7, dist * 0.85, dist * 0.7], fov: 45, near: 0.1, far: dist * 10 }} shadows>
        <color attach="background" args={['#eef2f6']} />
        <ambientLight intensity={0.75} />
        <directionalLight position={[dist, dist * 1.5, dist * 0.5]} intensity={1.1} castShadow />
        <directionalLight position={[-dist, dist, -dist]} intensity={0.35} />
        {/* plan XY → world XZ (floor), walls extrude up +Y; centre at origin */}
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <group position={[-center[0], -center[1], 0]}>
            {floors.map((f) => (
              <mesh key={f.id} geometry={f.geom} position={[0, 0, -0.01]}>
                <meshStandardMaterial color={f.color} side={THREE.DoubleSide} transparent opacity={0.85} />
              </mesh>
            ))}
            {wallGeom && (
              <mesh geometry={wallGeom} castShadow receiveShadow>
                <meshStandardMaterial color="#dfe4ea" side={THREE.DoubleSide} />
              </mesh>
            )}
          </group>
        </group>
        <OrbitControls makeDefault target={[0, 0, 0]} enableDamping dampingFactor={0.08} />
      </Canvas>
    </div>
  );
}
