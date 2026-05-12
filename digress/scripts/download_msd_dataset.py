import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET = "caspervanengelenburg/modified-swiss-dwellings"
DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "msd" / "raw"
DEFAULT_GRAPHS_OUT = PROJECT_ROOT / "data" / "msd" / "graphs.p"
DEFAULT_WALL_GRAPHS_OUT = PROJECT_ROOT / "data" / "msd_wall" / "graphs.p"
DEFAULT_CSV_NAME = "mds_V2_5.372k.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the Modified Swiss Dwellings dataset from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Raw dataset output directory. Default: ../datasets/msd/raw",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to Kaggle download even if local files already exist.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Keep the Kaggle zip instead of extracting it.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="After download, create digress/data/msd/graphs.p and sample visualizations.",
    )
    parser.add_argument(
        "--graphs-out",
        default=str(DEFAULT_GRAPHS_OUT),
        help="Prepared graph pickle path when --prepare is used.",
    )
    parser.add_argument(
        "--csv-source",
        default=None,
        help=(
            "CSV source for --add-wall-edges. Defaults to the Kaggle "
            f"{DEFAULT_CSV_NAME} file under --out-dir."
        ),
    )
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit for smoke tests.")
    parser.add_argument("--visualize", type=int, default=10)
    parser.add_argument("--num-vis", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--add-wall-edges",
        action="store_true",
        help="Pass --add-wall-edges to prepare_msd_graphs.py.",
    )
    parser.add_argument(
        "--wall-contact-eps",
        type=float,
        default=0.01,
        help="Room-wall contact tolerance when --add-wall-edges is used with CSV source.",
    )
    parser.add_argument(
        "--wall-room-pair-max-distance",
        type=float,
        default=0.15,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wall-min-contact-length",
        type=float,
        default=0.02,
        help="Minimum room-wall boundary contact length required for a wall edge candidate.",
    )
    parser.add_argument(
        "--wall-segment-gap",
        type=float,
        default=0.45,
        help="Maximum distance between two room-wall contact segments on the same wall (v3 default 0.45, v4 default 0.10).",
    )
    parser.add_argument(
        "--wall-version",
        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
        default="v3",
        help="Wall-edge inference algorithm version. See prepare_msd_graphs.py --wall-version.",
    )
    parser.add_argument(
        "--wall-room-distance-eps",
        type=float,
        default=0.5,
        help="v5 only: max room-room distance to consider as wall-pair candidate.",
    )
    parser.add_argument(
        "--wall-midline-samples",
        type=int,
        default=16,
        help="v5 only: number of midline samples for wall coverage check.",
    )
    parser.add_argument(
        "--wall-coverage-threshold",
        type=float,
        default=0.6,
        help="v5 only: min fraction of midline samples inside any WALL polygon.",
    )
    parser.add_argument(
        "--wall-touch-share-min-length",
        type=float,
        default=0.04,
        help="v5/v6 only: min shared boundary length when rooms directly touch (corner-kiss filter).",
    )
    parser.add_argument(
        "--wall-min-shared-arc-length",
        type=float,
        default=0.5,
        help="v6 only: min length of each room's boundary arc that faces the other room (filters diagonal corners).",
    )
    parser.add_argument(
        "--stats-out",
        default=None,
        help="Optional dataset statistics JSON path passed to prepare_msd_graphs.py.",
    )
    parser.add_argument(
        "--vis-out",
        default=None,
        help="Optional sample graph grid PNG path passed to prepare_msd_graphs.py.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Optional per-sample visualization directory passed to prepare_msd_graphs.py.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="MSD split to prepare for the DiGress baseline. Default: train",
    )
    parser.add_argument(
        "--kaggle-command",
        default=None,
        help="Optional Kaggle command prefix, e.g. 'kaggle' or 'uvx kaggle'.",
    )
    return parser.parse_args()


def command_prefix(user_command):
    if user_command:
        return user_command.split()
    if shutil.which("kaggle"):
        return ["kaggle"]
    if shutil.which("uvx"):
        return ["uvx", "kaggle"]
    raise RuntimeError(
        "Could not find Kaggle CLI. Install kaggle or uv, then authenticate Kaggle first."
    )


def find_split_graph_out(raw_dir, split):
    candidates = sorted(raw_dir.glob(f"**/{split}/graph_out"))
    if not candidates:
        candidates = sorted(raw_dir.glob("**/graph_out"))
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: len(list(path.glob("*.pickle"))))


def find_csv_source(raw_dir, user_csv_source=None):
    if user_csv_source:
        csv_path = Path(user_csv_source).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV source not found: {csv_path}")
        return csv_path

    direct = raw_dir / DEFAULT_CSV_NAME
    if direct.exists():
        return direct

    candidates = sorted(raw_dir.glob(f"**/{DEFAULT_CSV_NAME}"))
    if candidates:
        return candidates[0]

    candidates = sorted(raw_dir.glob("**/*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find MSD CSV under {raw_dir}")


def download_dataset(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_graph_out = find_split_graph_out(out_dir, args.split)
    if existing_graph_out is not None and not args.force:
        print(f"MSD raw data already exists: {existing_graph_out}")
        return out_dir

    cmd = [
        *command_prefix(args.kaggle_command),
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(out_dir),
    ]
    if not args.no_unzip:
        cmd.append("--unzip")
    if args.force:
        cmd.append("--force")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def prepare_graphs(args, raw_dir):
    if args.add_wall_edges:
        source = find_csv_source(raw_dir, args.csv_source)
        # Per-version default output: V3 keeps the legacy data/msd_wall path,
        # other versions get suffixed dirs so they don't overwrite each other.
        if args.wall_version == "v3":
            default_graphs_out = DEFAULT_WALL_GRAPHS_OUT
        else:
            default_graphs_out = (
                PROJECT_ROOT / "data" / f"msd_wall_{args.wall_version}" / "graphs.p"
            )
    else:
        source = find_split_graph_out(raw_dir, args.split)
        if source is None:
            raise FileNotFoundError(f"Could not find {args.split}/graph_out under {raw_dir}")
        default_graphs_out = DEFAULT_GRAPHS_OUT

    graphs_out = Path(args.graphs_out).resolve()
    if args.add_wall_edges and graphs_out == DEFAULT_GRAPHS_OUT.resolve():
        graphs_out = default_graphs_out.resolve()
    output_dir = graphs_out.parent
    stats_out = Path(args.stats_out).resolve() if args.stats_out else output_dir / "dataset_stats.json"
    vis_out = Path(args.vis_out).resolve() if args.vis_out else output_dir / "sample_graphs.png"
    vis_dir = Path(args.vis_dir).resolve() if args.vis_dir else output_dir / "vis"

    prepare_script = PROJECT_ROOT / "scripts" / "prepare_msd_graphs.py"
    cmd = [
        sys.executable,
        str(prepare_script),
        "--source",
        str(source),
        "--out",
        str(graphs_out),
        "--min-nodes",
        str(args.min_nodes),
        "--max-nodes",
        str(args.max_nodes),
        "--visualize",
        str(args.visualize),
        "--num-vis",
        str(args.num_vis),
        "--progress-every",
        str(args.progress_every),
        "--stats-out",
        str(stats_out),
        "--vis-out",
        str(vis_out),
        "--vis-dir",
        str(vis_dir),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.add_wall_edges:
        cmd.append("--add-wall-edges")
        cmd.extend(["--wall-version", args.wall_version])
        cmd.extend(["--wall-contact-eps", str(args.wall_contact_eps)])
        cmd.extend(["--wall-min-contact-length", str(args.wall_min_contact_length)])
        cmd.extend(["--wall-segment-gap", str(args.wall_segment_gap)])
        if args.wall_version in {"v5", "v6"}:
            cmd.extend(["--wall-room-distance-eps", str(args.wall_room_distance_eps)])
            cmd.extend(["--wall-midline-samples", str(args.wall_midline_samples)])
            cmd.extend(["--wall-coverage-threshold", str(args.wall_coverage_threshold)])
            cmd.extend(["--wall-touch-share-min-length", str(args.wall_touch_share_min_length)])
        if args.wall_version == "v6":
            cmd.extend(["--wall-min-shared-arc-length", str(args.wall_min_shared_arc_length)])

    print("Preparing DiGress graphs from:", source)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    raw_dir = download_dataset(args)
    if args.prepare:
        prepare_graphs(args, raw_dir)
    else:
        graph_out = find_split_graph_out(raw_dir, args.split)
        if graph_out is not None:
            print(f"Next: python scripts/prepare_msd_graphs.py --source {graph_out}")


if __name__ == "__main__":
    main()
