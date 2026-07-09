"use client";

import * as React from "react";
import { Canvas } from "@react-three/fiber";
import { Edges, Grid, OrbitControls } from "@react-three/drei";

import { buildPlanModel } from "@/lib/plan3d";
import type { Plan } from "@/lib/types";

interface Viewport3DProps {
  plan: Plan;
}

export function Viewport3D({ plan }: Viewport3DProps) {
  const model = React.useMemo(() => buildPlanModel(plan), [plan]);
  const distance = Math.max(18, model.extentMeters * 1.45);

  return (
    <Canvas
      camera={{ position: [distance * 0.72, distance * 0.62, distance * 0.72], fov: 42, near: 0.1, far: distance * 12 }}
      dpr={[1, 2]}
      className="h-full w-full"
    >
      <color attach="background" args={["#f0efeb"]} />
      <ambientLight intensity={1.15} />
      <directionalLight position={[model.extentMeters, model.extentMeters * 1.6, model.extentMeters * 0.7]} intensity={1.35} />
      <directionalLight position={[-model.extentMeters, model.extentMeters * 0.9, -model.extentMeters * 0.5]} intensity={0.5} />

      {model.base ? (
        <mesh position={model.base.center}>
          <boxGeometry args={model.base.size} />
          <meshStandardMaterial color={model.base.color} />
        </mesh>
      ) : null}

      {model.floors.map((box, index) => (
        <mesh key={`floor-${index}`} position={box.center}>
          <boxGeometry args={box.size} />
          <meshStandardMaterial color={box.color} />
        </mesh>
      ))}

      {model.walls.map((box, index) => (
        <mesh key={`wall-${index}`} position={box.center}>
          <boxGeometry args={box.size} />
          <meshStandardMaterial color={box.color} roughness={0.85} />
        </mesh>
      ))}

      {model.doors.map((door, index) => (
        <React.Fragment key={`door-${index}`}>
          {/* Leaf: a thin panel swung 30° open, with a subtle edge outline. */}
          <mesh position={door.leaf.center} rotation={[0, door.leaf.rotationY, 0]}>
            <boxGeometry args={door.leaf.size} />
            <meshStandardMaterial color={door.leaf.color} roughness={0.6} />
            <Edges color="#b9b4a8" />
          </mesh>
          {/* Jamb posts framing the opening ends. */}
          {door.posts.map((post, postIndex) => (
            <mesh key={postIndex} position={post.center}>
              <boxGeometry args={post.size} />
              <meshStandardMaterial color={post.color} roughness={0.8} />
            </mesh>
          ))}
        </React.Fragment>
      ))}

      <Grid
        position={[0, -0.19, 0]}
        args={[model.extentMeters * 6, model.extentMeters * 6]}
        cellSize={2}
        cellColor="#d6d3ca"
        sectionSize={10}
        sectionColor="#c2beb2"
        fadeDistance={distance * 3.2}
        infiniteGrid
      />
      <OrbitControls makeDefault maxPolarAngle={Math.PI / 2.05} minDistance={4} maxDistance={distance * 4} />
    </Canvas>
  );
}
