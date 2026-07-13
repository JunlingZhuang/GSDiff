# Pixel Plan AI

Pixel Plan AI converts an architectural program list into a color-coded semantic floor plan. Gemini acts as a focused coding agent: it writes a complete Python program, executes it in an isolated child process, inspects deterministic architectural checks, and revises the complete program when execution or validation fails.

Healthcare generation uses the versioned `us-healthcare-schematic-v1` rules profile. It contains U.S. schematic-planning defaults for room area, minimum clear dimension, aspect ratio, compactness, functional access, and patient-room suite relationships. It is not a substitute for the adopted FGI edition, state requirements, accessibility review, or authority-having-jurisdiction approval.

Generation is cancellable. The canvas displays an elapsed-time overlay while a job is active, and Stop Generation terminates both the browser request and its backend worker process. After a successful result, the main action becomes Re-generate Plan.

## Stack

- Backend: Python 3.11 plus pinned scientific and geometry libraries
- Frontend source: TypeScript with no runtime framework
- Frontend build dependency: TypeScript 5.4.2 only
- AI provider: Google Gemini only
- Generated-code runtime: broad Python with approved imports and a fixed `result` data contract

The browser cannot execute TypeScript directly. `frontend/app.ts` is the source of truth; `public/app.js` is its generated browser artifact. No JavaScript source is maintained manually.

## Configure Gemini

Copy `.env.example` to `.env` in the project root and set:

```env
GEMINI_API_KEY=your_key_here
GEMINI_FAST_MODEL=gemini-3.1-flash-lite
GEMINI_MODEL=gemini-3.5-flash
GEMINI_FAST_THINKING_LEVEL=minimal
GEMINI_THINKING_LEVEL=low
GEMINI_REPAIR_THINKING_LEVEL=low
GEMINI_MAX_ATTEMPTS=5
GEMINI_TIMEOUT_SECONDS=300
GENERATION_TIMEOUT_SECONDS=330
PORT=8765
```

The API key is loaded only by the Python backend. It is never sent to the browser or included in generated plans.

The default route starts with `gemini-3.1-flash-lite` at minimal thinking. A validation-driven repair uses `gemini-3.5-flash` at low thinking. This keeps the common path fast while the validator prevents malformed plans from being accepted. The Run Log records the model used for every attempt.

## Run on Windows

One command starts the full stack (kills stale instances, builds the web app on
first launch, opens three titled service windows, probes health, then opens the
browser):

```powershell
powershell -ExecutionPolicy Bypass -File pixel-plan-ai\start-all.ps1
```

Or start the three services manually, one terminal each:

```powershell
# 1) Python backend - generation / validation / repair loop (:8765, required)
cd pixel-plan-ai
python -B backend\server.py

# 2) hfagent service - candidate drawing + tracing (:8801, optional;
#    Image/Trace modes degrade to fallbacks without it).
#    MUST use hfagent's own venv (google-genai + cv2 live there):
cd <repo root>
hfagent\.venv\Scripts\python -m hfagent.tools.trace_server

# 3) Studio frontend (:3000). `next start` serves the last build;
#    run `npm run build` after frontend changes, or use `npx next dev` while developing.
cd pixel-plan-ai\web
npx next start -p 3000
```

Open `http://127.0.0.1:3000` (the Studio). Notes:

- Backend Python changes do NOT need a server restart - every generation runs in
  a fresh worker subprocess. Only `backend/server.py` route changes need one.
- If new routes 404 after a restart, kill every stale `server.py` process first:
  Windows lets several instances bind the same port, and a stale one keeps
  answering with old code (`start-all.ps1` does this cleanup for you).
- The legacy vanilla-TS UI is still served by the backend at `http://127.0.0.1:8765`
  (falls back to `8787` when busy); `.\start.ps1` starts only that.

## Rebuild the TypeScript frontend

The compiled artifact is included. Rebuild it after changing `frontend/app.ts`:

```powershell
tsc -p tsconfig.json
```

If `tsc` is not installed globally:

```powershell
npm install
npm run build
```

`package.json` pins the only frontend dependency, `typescript`.

## Run tests

```powershell
npm test
```

The tests cover the NumPy result contract, door-boundary checks, manual inspection, live progress checkpoints, relaxed area and proportion thresholds, revision context, and the five-attempt repair loop.

## Repair loop and Run Log

Gemini acts as a floor-plan coding agent in a bounded executor-feedback loop:

1. Write a complete Python layout algorithm.
2. Execute it in the restricted child process and validate the resulting plan.
3. Send the exact runtime or validation failure plus the previous complete program back to Gemini.
4. Repeat until a candidate passes, up to `GEMINI_MAX_ATTEMPTS` (maximum 5).
5. If no candidate passes, return the highest-scoring executable Gemini candidate. If none executes, report an explicit generation failure.

Every attempt records its phase, source, status, live duration, validation score, issues, complete code, token usage, and estimated API cost. A `running` entry appears before the Gemini request begins; completed checkpoints update Run Log and Generated Code while later attempts are still running.

The Generated Code tab keeps the active complete program as editable state:

- `Run Code` executes the edited program.
- `Inspect` executes it and opens deterministic architectural results.
- `Fix with AI` sends the complete program and exact failures back to Gemini.
- `Revise with AI` sends the complete program plus a user instruction to Gemini.

## Structured AI output

Gemini is constrained with a JSON response schema:

```json
{
  "code": "import numpy as np\n...\nresult = {...}",
  "strategy": "A concise layout strategy",
  "assumptions": ["An explicit assumption"]
}
```

The generated code is ordinary Python. It may use functions, classes, loops, exceptions, recursion, comprehensions, and its own layout or optimization algorithms. Approved imports are:

```text
numpy, Pillow, shapely, scipy, networkx, math, random,
statistics, itertools, collections
```

The only required programming contract is a top-level `result` dictionary:

```python
result = {
    "width": width,
    "height": height,
    "meters_per_cell": meters_per_cell,
    "footprint": footprint,  # height x width boolean array
    "grid": grid,            # room indexes, -1 unassigned, -2 outside
    "rooms": rooms,          # unique id and program type
    "doors": doors,          # exact shared-boundary coordinates; swing is derived after execution
}
```

File access, network access, processes, dynamic execution, unsafe imports, and private runtime attributes are blocked. Accepted code runs in a short-lived child process with a 12-second timeout and an output-size limit. The legacy `PixelPlan` runtime is retained only to execute older saved programs; it is not a generator or fallback, and Gemini is explicitly instructed not to use it.

After execution, the normalizer derives each door's `swing_side` from the final grid. Corridor-room doors swing into the room, and exterior entrances swing into the building; the AI does not guess this direction.

## U.S. healthcare rules

The structured knowledge base is stored in `data/healthcare_rules_us.json`. It is included in the Gemini context and enforced again by the deterministic validator. Rules currently cover:

- target-area tolerance and minimum net square feet;
- minimum clear room dimension in feet;
- preferred and maximum room aspect ratio;
- minimum rectangular compactness;
- direct corridor access by room type;
- independent support rooms versus allowed patient-room toilet suites;
- U.S. customary area input through `approx_area_ft2`.

When both `approx_area_ft2` and `approx_area_m2` are present, square feet is authoritative and converted internally to square meters for raster calculations.

## Project layout

```text
backend/              Python API, Gemini client, code policy, runtime, validation
data/samples.json     Supplied program-list samples
data/healthcare_rules_us.json  Versioned U.S. schematic healthcare rules
frontend/app.ts       TypeScript frontend source
public/               Static UI and compiled TypeScript output
tests/                Python tests for policy and generation
.env                   Local secret configuration, ignored by Git
```

## Scope

This prototype produces semantic planning studies. Its output is not construction documentation and does not yet verify accessibility, egress, building code, structural systems, or clinical regulations.
