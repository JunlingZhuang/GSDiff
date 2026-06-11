# hfagent — Healthcare Floor Plan Agent (backend)

Phase 0 of [docs/agent/](../docs/agent/README.md). Self-contained: own venv, no
dependency on the GSDiff ML code or `app/`.

## Layout

```
hfagent/
├── llm.py             # ONLY place that talks to Gemini (API-key or Vertex, env-driven)
├── metrics.py         # plan-vs-plan IoU / count / type metrics
├── schema/
│   ├── plan.py        # Plan/Room (+ Wall/Door placeholders), mm units, versioned
│   └── palette.py     # room-type colour legend shared by ALL tools
├── tools/             # the three pipeline tools (future LangGraph nodes)
│   ├── generate_colorblock.py   # ① program -> Gemini -> colour-block PNG
│   ├── cv_parse.py              # ② PNG -> Plan JSON (OpenCV, deterministic)
│   └── render_plan.py           # ③ Plan JSON -> PNG (deterministic re-render)
├── tests/             # pytest: contracts + round-trip gate (no API key needed)
├── docs/              # findings reports (phase0a-findings.md, illustrated)
├── out/               # real-VLM sweep outputs (gemini/parsed/recon per program)
└── run_phase0a.py     # real-VLM evaluation sweep (needs GEMINI key)
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

## Phase 0-A: two-layer validation

```bash
pytest tests/                      # layer 1: round-trip gate (deterministic, no VLM)
python -m hfagent.run_phase0a      # layer 2: real Gemini sweep -> hfagent/out/phase0a/
```

Layer 1 proves the parser on synthetic renders (known ground truth); layer 2
then measures pure VLM image quality with the already-trusted parser.
