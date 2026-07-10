// Parametric GLB builder for the ICU room asset library.
//
// Produces one binary .glb per catalog type into web/public/assets/3d/. The
// runtime (RoomViewport3D) loads these directly through drei useGLTF and never
// runs this script — the generated files are committed. Re-run only when the
// geometry recipe changes:
//
//   node scripts/generate-3d-assets.mjs
//
// Units are meters at real-world scale (1 ft = 0.3048 m). Each asset's local
// origin is the CENTER of its footprint on the floor plane (y = 0 is the floor,
// +y is up), so the viewer can drop it at a room (x, z) and spin it about its
// own centre. The one exception is ceiling_boom: its origin sits at the ceiling
// mount plate (y = 0) and the geometry hangs downward (y < 0), so the viewer
// positions it at ceiling height.
//
// FACING CONVENTION (must match the 2D symbols in web/public/assets/2d and the
// backend validator's WALL_FACING_ROTATION map):
//   At rotation_deg 0 every asset's FRONT is its SOUTH edge. Room y (north) maps
//   to world +z in the viewer (RoomViewport3D), so SOUTH = local -z. Author each
//   builder so its front feature (casework doors, monitor screen, sink basin
//   approach, chair seat opening) points to local -z and its wall-side back
//   (backsplash, monitor bracket, sink faucet, chair backrest) points to +z.
//   The viewer then rotates a wall-anchored asset by wall so the front turns
//   into the room: N->0, S->180, E->270, W->90 (90/270 swap the placed
//   footprint). icu_bed is exempt: its head/foot axis is set by layout logic,
//   not by this front-south rule.

import * as THREE from "../web/node_modules/three/build/three.module.js";
import { GLTFExporter } from "../web/node_modules/three/examples/jsm/exporters/GLTFExporter.js";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// GLTFExporter's binary path reads a Blob through a browser FileReader. Node has
// Blob but not FileReader, so bridge it with Blob.arrayBuffer() (geometry-only
// export never touches image/canvas paths, so this minimal shim is sufficient).
if (typeof globalThis.FileReader === "undefined") {
  globalThis.FileReader = class {
    readAsArrayBuffer(blob) {
      Promise.resolve(blob.arrayBuffer()).then(
        (buffer) => {
          this.result = buffer;
          this.onloadend?.();
        },
        (error) => this.onerror?.(error),
      );
    }
  };
}

const F = 0.3048; // feet -> meters

// Consistent materials: matte off-white bodies, warm-grey accents. The entrance
// accent is reserved for door/entry treatment and intentionally unused here.
const BODY = new THREE.MeshStandardMaterial({ color: 0xf4f2ed, roughness: 0.85, metalness: 0.0 });
const ACCENT = new THREE.MeshStandardMaterial({ color: 0xc9c4ba, roughness: 0.7, metalness: 0.05 });
const ENTRANCE_ACCENT = new THREE.MeshStandardMaterial({ color: 0x8fbaf0, roughness: 0.6 }); // reserved

function box(wFt, hFt, dFt, xFt, yFt, zFt, material, rotationYDeg = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(wFt * F, hFt * F, dFt * F), material);
  mesh.position.set(xFt * F, yFt * F, zFt * F);
  if (rotationYDeg) mesh.rotation.y = (rotationYDeg * Math.PI) / 180;
  return mesh;
}

function cylinder(radiusFt, heightFt, xFt, yFt, zFt, material) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radiusFt * F, radiusFt * F, heightFt * F, 20),
    material,
  );
  mesh.position.set(xFt * F, yFt * F, zFt * F);
  return mesh;
}

// Each builder returns the meshes that make up one asset (footprint-centred).
const builders = {
  // Platform + mattress + raised head deck + rails. Head is at -z.
  icu_bed() {
    return [
      box(3.4, 0.45, 7.4, 0, 0.35, 0, BODY), // frame
      box(3.15, 0.3, 6.9, 0, 0.72, 0, ACCENT), // mattress
      box(3.15, 0.18, 1.6, 0, 0.95, -2.7, ACCENT), // raised head deck
      box(3.5, 1.1, 0.18, 0, 0.75, -3.72, BODY), // headboard
      box(3.5, 0.72, 0.18, 0, 0.6, 3.72, BODY), // footboard
      box(0.12, 0.5, 3.6, 1.72, 0.95, 0.2, BODY), // right rail
      box(0.12, 0.5, 3.6, -1.72, 0.95, 0.2, BODY), // left rail
    ];
  },
  // Ceiling plate + vertical drop + two-segment arm + equipment head; hangs from y = 0.
  ceiling_boom() {
    return [
      box(1.2, 0.15, 1.2, 0, -0.075, 0, BODY), // ceiling plate
      box(0.22, 1.2, 0.22, 0, -0.75, 0, BODY), // vertical drop
      box(1.3, 0.16, 0.16, 0.45, -1.35, 0, ACCENT), // arm segment 1
      box(0.16, 0.16, 1.0, 1.05, -1.35, 0.4, ACCENT), // arm segment 2
      box(0.8, 0.9, 0.6, 1.05, -1.85, 0.85, BODY), // equipment head
    ];
  },
  // Thin screen on a short wall bracket, mounted high. Bracket is the wall-side
  // back at +z; the screen face is the FRONT at -z (south) so it faces the room.
  patient_monitor() {
    return [
      box(0.25, 0.25, 0.35, 0, 1.45, 0.18, ACCENT), // bracket (back, +z)
      box(1.5, 1.0, 0.14, 0, 1.5, -0.05, BODY), // screen body
      box(1.3, 0.8, 0.03, 0, 1.5, -0.13, ACCENT), // screen face (front, -z)
    ];
  },
  // Pole + hooks + 5-leg base (approximated with radiating feet).
  iv_pole() {
    const meshes = [
      cylinder(0.035, 5.2, 0, 2.6, 0, BODY), // pole
      cylinder(0.18, 0.1, 0, 0.14, 0, ACCENT), // hub
      box(0.35, 0.05, 0.06, 0.2, 5.0, 0, ACCENT), // hook arm x
      box(0.06, 0.05, 0.35, 0, 5.0, 0.2, ACCENT), // hook arm z
    ];
    for (let i = 0; i < 5; i += 1) {
      const angle = (i * 72 * Math.PI) / 180;
      meshes.push(box(0.6, 0.07, 0.11, 0.32 * Math.cos(angle), 0.06, -0.32 * Math.sin(angle), ACCENT, i * 72));
    }
    return meshes;
  },
  // Seat + backrest + arms + legs. Backrest is the back at +z; the seat opening
  // (front) faces -z (south) so the sitter faces the room.
  visitor_chair() {
    return [
      box(1.7, 0.22, 1.6, 0, 1.35, -0.1, BODY), // seat
      box(1.7, 1.5, 0.2, 0, 2.1, 0.8, BODY), // backrest (back, +z)
      box(0.2, 0.6, 1.5, 0.82, 1.75, -0.05, ACCENT), // right arm
      box(0.2, 0.6, 1.5, -0.82, 1.75, -0.05, ACCENT), // left arm
      box(0.15, 1.35, 0.15, 0.72, 0.675, -0.72, ACCENT),
      box(0.15, 1.35, 0.15, -0.72, 0.675, -0.72, ACCENT),
      box(0.15, 1.35, 0.15, 0.72, 0.675, 0.72, ACCENT),
      box(0.15, 1.35, 0.15, -0.72, 0.675, 0.72, ACCENT),
    ];
  },
  // Carcass + counter overhang + door panels + pulls. Doors are the FRONT and
  // face -z (south) into the room; the wall-side back is +z.
  casework() {
    const meshes = [
      box(5.9, 2.9, 1.9, 0, 1.45, 0, BODY), // carcass
      box(6.2, 0.15, 2.2, 0, 3.0, -0.15, ACCENT), // counter top (overhangs the front)
    ];
    for (const x of [-1.9, 0, 1.9]) {
      meshes.push(box(1.7, 2.4, 0.06, x, 1.4, -0.98, ACCENT)); // door panel (front, -z)
      meshes.push(box(0.09, 0.35, 0.09, x + 0.7, 1.9, -1.0, BODY)); // pull
    }
    return meshes;
  },
  // Cabinet + counter + basin recess (boolean-free inset) + faucet. The faucet
  // is the wall-side back at +z; the basin approach (front) faces -z (south).
  handwash_sink() {
    return [
      box(1.9, 2.85, 1.65, 0, 1.425, 0, BODY), // cabinet
      box(2.1, 0.15, 1.85, 0, 3.0, 0, ACCENT), // counter
      box(1.1, 0.45, 0.9, 0, 2.72, -0.05, ACCENT), // basin recess (front side)
      box(0.1, 0.55, 0.1, 0, 3.3, 0.6, BODY), // faucet post (back, +z)
      box(0.1, 0.1, 0.45, 0, 3.55, 0.42, BODY), // faucet spout
    ];
  },
  // Top + offset column + C-base.
  overbed_table() {
    return [
      box(2.5, 0.14, 1.25, 0, 2.7, 0, BODY), // top
      box(0.22, 2.6, 0.22, -0.9, 1.35, 0, ACCENT), // column
      box(0.3, 0.12, 1.1, -0.9, 0.06, 0, ACCENT), // foot
      box(0.9, 0.12, 0.25, -0.5, 0.06, 0, ACCENT), // foot connector
    ];
  },
};

void ENTRANCE_ACCENT; // reserved material kept referenced

async function exportGlb(object) {
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(
      object,
      (result) => resolve(Buffer.from(result)),
      (error) => reject(error),
      { binary: true, onlyVisible: true },
    );
  });
}

async function main() {
  const outDir = fileURLToPath(new URL("../web/public/assets/3d/", import.meta.url));
  await mkdir(outDir, { recursive: true });

  for (const [type, builder] of Object.entries(builders)) {
    const group = new THREE.Group();
    group.name = type;
    for (const mesh of builder()) group.add(mesh);
    const buffer = await exportGlb(group);
    const path = `${outDir}${type}.glb`;
    await writeFile(path, buffer);
    console.log(`${type}.glb  ${buffer.length.toLocaleString()} bytes  (${group.children.length} meshes)`);
  }
  console.log("done.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
