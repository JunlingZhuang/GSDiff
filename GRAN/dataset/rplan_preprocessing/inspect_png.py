"""Inspect raw RPLAN PNG(s) to verify channel structure and room/door encoding.

Channel layout (verified against https://github.com/zzilch/RPLAN-Toolbox):
    Channel 0: boundary   - building boundary; pixel 255 = front door location
    Channel 1: category   - room types + walls + doors (value IDs below)
    Channel 2: instance   - per-room unique instance ID
    Channel 3: inside     - interior mask (255=inside, 0=exterior)

Channel 1 category value IDs (verified against RPLAN-Toolbox):
      0 LivingRoom       9 Balcony        14 ExteriorWall
      1 MasterRoom      10 Entrance       15 FrontDoor
      2 Kitchen         11 Storage        16 InteriorWall
      3 Bathroom        12 Wall-in        17 InteriorDoor
      4 DiningRoom      13 External
      5 ChildRoom
      6 StudyRoom
      7 SecondRoom
      8 GuestRoom

Usage:
    # Randomly sample N files from a directory (default: 10):
    uv run python dataset/rplan_preprocessing/inspect_png.py \\
        --dir D:/Github/GSDiff/datasets/rplandata/Data/floorplan_dataset \\
        --out_dir inspect_out/

    # Single file:
    uv run python dataset/rplan_preprocessing/inspect_png.py \\
        --png D:/Github/GSDiff/datasets/rplandata/Data/floorplan_dataset/0.png
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CATEGORY_NAMES = {
    0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
    4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
    8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage',
    12: 'Wall-in', 13: 'External',
    14: 'ExteriorWall', 15: 'FrontDoor',
    16: 'InteriorWall', 17: 'InteriorDoor',
}

ROOM_VALUES = set(range(13))           # 0-12 are room types
WALL_VALUES = {14, 16}                 # ExteriorWall, InteriorWall
DOOR_VALUES = {15, 17}                 # FrontDoor, InteriorDoor
INTERIOR_DOOR = 17
FRONT_DOOR = 15


def inspect(png_path, out_path):
    if not os.path.isfile(png_path):
        print(f'Error: {png_path} not found')
        sys.exit(1)

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'Error: cannot read {png_path}')
        sys.exit(1)

    print(f'Image shape: {img.shape}  dtype: {img.dtype}')

    if img.ndim != 3 or img.shape[2] != 4:
        print(f'Warning: expected 4 channels, got shape {img.shape}')
        return

    boundary = img[:, :, 0]
    category = img[:, :, 1]
    instance = img[:, :, 2]
    inside = img[:, :, 3]

    print('\n-- Channel 0 (boundary) --')
    unique_b = sorted(np.unique(boundary).tolist())
    print(f'  unique values: {unique_b}')
    print(f'  255 (front door marker) pixels: {int((boundary == 255).sum())}')

    print('\n-- Channel 1 (category) --')
    unique_c = sorted(np.unique(category).tolist())
    print('  value distribution:')
    for v in unique_c:
        count = int((category == v).sum())
        name = CATEGORY_NAMES.get(v, f'unknown({v})')
        print(f'    {v:3d} ({name:14s}): {count:6d} pixels')

    room_pixels = int(np.isin(category, list(ROOM_VALUES)).sum())
    wall_pixels = int(np.isin(category, list(WALL_VALUES)).sum())
    interior_door_pixels = int((category == INTERIOR_DOOR).sum())
    front_door_pixels = int((category == FRONT_DOOR).sum())
    print(f'\n  Total room pixels (0-12): {room_pixels}')
    print(f'  Total wall pixels (14,16): {wall_pixels}')
    print(f'  Interior door pixels (17): {interior_door_pixels}')
    print(f'  Front door pixels (15):    {front_door_pixels}')

    print('\n-- Channel 2 (instance) --')
    unique_i = sorted(np.unique(instance).tolist())
    print(f'  unique instances: {unique_i[:20]}'
          f'{"..." if len(unique_i) > 20 else ""}')
    print(f'  total distinct instances: {len(unique_i)}')

    print('\n-- Channel 3 (inside mask) --')
    print(f'  min={inside.min()} max={inside.max()} '
          f'inside pixels={int((inside > 0).sum())}')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(boundary, cmap='gray')
    axes[0, 0].set_title('Ch0: boundary (255=front door)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(category, cmap='tab20', vmin=0, vmax=17)
    axes[0, 1].set_title('Ch1: category (rooms + walls + doors)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(instance, cmap='tab20')
    axes[0, 2].set_title('Ch2: instance ID')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(inside, cmap='gray')
    axes[1, 0].set_title('Ch3: inside mask')

    wall_mask = np.isin(category, list(WALL_VALUES))
    door_mask = np.isin(category, list(DOOR_VALUES))
    overlay = category.copy().astype(np.float32)
    overlay[wall_mask] = np.nan
    overlay[door_mask] = np.nan
    im4 = axes[1, 1].imshow(overlay, cmap='tab20', vmin=0, vmax=12)
    axes[1, 1].set_title('Rooms only (walls/doors blanked)')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    structure = np.zeros_like(category, dtype=np.uint8)
    structure[wall_mask] = 1
    structure[category == INTERIOR_DOOR] = 2
    structure[category == FRONT_DOOR] = 3
    axes[1, 2].imshow(structure, cmap='tab10', vmin=0, vmax=4)
    axes[1, 2].set_title('Structure: wall(1), int door(2), front door(3)')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'RPLAN PNG inspection: {os.path.basename(png_path)}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--png', type=str, default=None,
                        help='Path to a single RPLAN PNG file')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing RPLAN PNG files')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of random samples (default: 10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    if args.png is None and args.dir is None:
        parser.error('Must specify either --png or --dir')

    os.makedirs(args.out_dir, exist_ok=True)

    if args.png is not None:
        inspect(args.png, os.path.join(args.out_dir, 'inspect_000.png'))
        return

    all_pngs = sorted(glob.glob(os.path.join(args.dir, '*.png')))
    if not all_pngs:
        print(f'No PNGs in {args.dir}')
        sys.exit(1)

    rng = random.Random(args.seed)
    sampled = rng.sample(all_pngs, min(args.num, len(all_pngs)))
    print(f'Found {len(all_pngs)} PNGs, sampling {len(sampled)} '
          f'(seed={args.seed})\n')

    for idx, png in enumerate(sampled):
        stem = os.path.splitext(os.path.basename(png))[0]
        out = os.path.join(args.out_dir, f'inspect_{idx:03d}_{stem}.png')
        print(f'\n[{idx+1}/{len(sampled)}] {png}')
        print('=' * 60)
        try:
            inspect(png, out)
        except Exception as e:
            print(f'  ERROR: {e}')


if __name__ == '__main__':
    main()
