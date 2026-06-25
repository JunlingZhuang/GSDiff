# hfagent — Healthcare Floor Plan Agent (backend)

Phase 0 of [docs/agent/](../docs/agent/README.md). Self-contained: own venv, no
dependency on the GSDiff ML code or `app/`.

## Layout

```
hfagent/
├── llm.py                  # ONLY place that talks to Gemini (API-key or Vertex, env-driven)
├── floor_plan_generate.py  # orchestrator: program -> correction loop -> plan + room graph
├── graph.py / api.py       # understand -> generate_plan wrappers (FastAPI route)
├── evaluate.py             # real-Gemini evaluation sweep over programs.json (needs key)
├── programs.json           # test room programs (one per building type)
├── config.json             # generation_mode (real2color) + max_correction_rounds
├── nodes/
│   └── understand.py       # brief -> structured program (always-run LLM node)
├── schema/
│   ├── plan.py             # Plan / Room (mm units, versioned)
│   ├── roomgraph.py        # RoomGraph: rooms as nodes, doors as edges
│   └── palette.py          # room-type colour legend shared by ALL tools
├── tools/
│   ├── floor_plan_generator.py     # real2color: text->real plan->colour-block + room graph
│   ├── room_adjacency_extractor.py # doors read from the realistic plan image
│   ├── image_parser.py             # colour-block PNG -> Plan (OpenCV, deterministic)
│   ├── render_plan.py              # Plan -> PNG + door overlay (deterministic)
│   └── plan_fixes.py               # deterministic room-count repair
├── tests/                  # pytest: contracts + round-trip + per-program smoke (no key)
├── docs/                   # findings reports (illustrated)
└── out/                    # outputs: out/tests/, out/eval/<ts>/, out/api/<ts>/
```

**Phase 0-A result: route viable** — see [docs/phase0a-findings.md](docs/phase0a-findings.md).

## Setup

```bash
cd hfagent
uv venv .venv --python 3.11
source .venv/Scripts/activate          # Windows bash
uv pip install -r requirements.txt
```

Credentials: `GEMINI=<api key>` in repo-root `.env.local` (already there), or
Vertex via `GOOGLE_GENAI_USE_VERTEXAI=true` + `GOOGLE_CLOUD_PROJECT`. Model ids
auto-resolve to the newest available (override: `HFAGENT_TEXT_MODEL`,
`HFAGENT_IMAGE_MODEL`).

## Testing

Run all commands from the **repo root** (`D:\Github\GSDiff`).

### 1. Unit / contract tests — deterministic, no API key

```bash
python -m pytest hfagent/tests/ -q          # full suite
python -m pytest hfagent/tests/test_programs.py -v   # every programs.json entry (mock VLM)
```

Mock tests prove wiring only. Artifacts persist to `hfagent/out/tests/<test_name>/`
(via the `out_dir` fixture) so the generated images are inspectable.

### 2. Full real-Gemini evaluation — REQUIRED after any pipeline change

Mock tests do **not** exercise model behaviour. After any change to the generation
pipeline (real2color flow, correction loop, prompts, room-adjacency extraction,
parsing, rendering, schemas), run the **full** sweep — never just `--n 1`:

```bash
python -m hfagent.evaluate                       # ALL programs, real Gemini -> out/eval/<timestamp>/
python -m hfagent.evaluate --program ward-wing   # one program by name
python -m hfagent.evaluate -p clinic-small,ward-wing   # a few, comma-separated
```

Then inspect, per program: `gemini_r*.real.png` (realistic plan with doors),
`gemini_r*.png` (colour-block), `recon_with_door.png`, `plan.json` (authoritative
model), `graph.json`, and the top-level `summary.json`. Requires `GEMINI=<key>` in
repo-root `.env.local`.

Useful flags: `--program` / `-p` / `--only <name>[,<name>]` (run named program(s)),
`--n <N>` (first N — quick smoke only, **not** a substitute for the full run),
`--out <dir>` (override the timestamped dir).
