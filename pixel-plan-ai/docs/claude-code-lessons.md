# Repair-loop roadmap: lessons from reading Claude Code's source

Status: proposal (nothing below is implemented yet). 2026-07-08.

Source of the lessons: a firsthand read of the leaked Claude Code TypeScript
source (`yasasbanukaofficial/claude-code`, ~1,900 files, recovered from npm
sourcemaps). Authenticity was verified by matching its prompt-assembly output
against a live Claude Code session word for word. The repo is unlicensed
proprietary code: we mine patterns, we never copy code.

## Where we stand architecturally

Pixel Plan AI is **not an agent — it is a deterministic workflow with an LLM
inside**, and that is a strength, not a gap. Control flow lives entirely in
`backend/service.py` (`generate -> sandbox execute -> validate -> repair
(<=5) -> stop`); the model has exactly one action (emit a complete program)
and never chooses what to do next or when to stop. In the Building Effective
Agents taxonomy this is an *evaluator-optimizer workflow*. Claude Code is a
true agent: its model owns the control flow (picks tools, decides
termination), which buys flexibility at the price of unbounded cost and
unpredictable paths. Our domain has a fixed known path and a deterministic
validator — the workflow is the right architecture. The one agent-shaped
experiment worth running is #8 below.

Things Claude Code does that we already match or beat, so no action:

- **History compaction** — our repair context only ever carries the latest
  program, which is what their message-splicing achieves.
- **Bounded retries** — their structured-output retry ceiling is 5, same as
  our repair loop; the designs are isomorphic.
- **Per-agent prompt differentiation** — our three mode guidance slots
  (`layout_family_guidance` / `REFERENCE_IMAGE_GUIDANCE` /
  `SEED_REPAIR_GUIDANCE`) are exactly their per-subagent prompt pattern.
- **Verification** — our deterministic architectural validator is a stronger
  verifier than their test-running + verification-agent stack.

## A. Repair accuracy (one coherent batch)

### 1. Incremental validator feedback (delta)

The Gemini API is stateless: every attempt is a fresh `generateContent` call,
and the model's only knowledge of previous attempts is whatever we paste into
`repair_context`. Today that is an isolated snapshot (full previous program +
full validation report), so the model cannot distinguish:

1. failures it already fixed (should not touch),
2. failures that persist (should focus),
3. failures its last edit *introduced* (should revert).

Point 3 is the killer, and we have hard evidence it happens: the
`best_candidate` rollback logic in `generate_plan` exists precisely because
candidates regress.

Design: keep `previous_validation` in the loop state, compute
`validation_delta(prev, cur)` in Python (the model needs no memory), and
prepend it to the repair feedback:

```
[validator delta attempt=3 vs attempt=2]
fixed: doors:exam_room(3/3), area:toilet
still_failing: count:waiting 0/1 · adjacency:waiting<->corridor
new_failures: none
```

Full detail is kept only for `still_failing` and `new_failures`; `fixed`
items shrink to one line. The complete current program is still sent — the
model cannot edit code it cannot see. This improves attempts 2-5 (repair
convergence), not attempt 1 (spatial reasoning is unchanged); the biggest
wins are trace-mode seed repairs, tower-scale programs, and runs that
currently exhaust 5 attempts.

Touches: `service.py` (one new function + loop bookkeeping).

### 2. Stable tags on validator errors

Claude Code shapes every tool error as `<tool_use_error>TypedPrefix:
...</tool_use_error>` so the model can pattern-match across turns. Our
feedback lines are free prose that the model re-parses every attempt.

Design: keep `validator.py` untouched; in `compact_validation_feedback`, wrap
issues with stable keys derived from existing check categories + room ids:

```
<validator_error rule="doors" room="waiting_0">no door on any corridor boundary; add one on the shared run at y=14, x in [2,20]</validator_error>
<validator_error rule="area" room="exam_room_2">actual 61 sf / 9 cells, target 120 sf +/- 35%; widen by ~1 cell</validator_error>
```

`rule:room` doubles as the comparison key for #1's delta.

Touches: `compact_validation_feedback` only.

### 3. Two behavioral rules in SYSTEM_INSTRUCTION

Concrete, decidable prompt rules (their style — no slogans):

- Verdict ownership: `The validator is the sole judge of correctness. Never
  state or imply that the plan passes; report what you changed and which
  named failures it targets.`
- Anti-thrashing: `Before changing approach, diagnose WHY the previous
  attempt failed from the validator delta. Prefer the minimal edit that
  fixes the named failures; never discard parts that already pass.`

The second rule is the usage discipline for #1's signal; ship them together.

Touches: two lines in `gemini.py`.

**Verification protocol for batch A**: fixed benchmark set (several programs
including a tower and 2-3 trace-mode seeds), A/B against main. Metrics:
acceptance rate, mean attempts, regression count (best-candidate rollback
triggers). Adopt only on a data win.

## B. Cost

### 4. Static/dynamic prompt layering

Claude Code splits its system prompt at an explicit
`SYSTEM_PROMPT_DYNAMIC_BOUNDARY`: a byte-stable cacheable prefix, then
per-session content, with comments warning that each runtime conditional
before the boundary doubles the cache-prefix variants. Gemini's implicit
caching is also prefix-based.

Design: reorder `build_prompt` into three layers — (i) globally static
(role, execution environment, result contract, grid contract, layout
requirements, healthcare profile), (ii) program-level (program JSON, pixel
targets, scale, mode guidance slot), (iii) attempt-level (repair context,
already last). Content unchanged, order only. In a 5-attempt run, attempts
2-5 then hit the cached static prefix (~60% of prompt tokens).

Touches: `build_prompt` section moves; requires a three-mode regression run.

### 5. Budget guard + typed stop reasons

Claude Code checks cumulative cost after every message and terminates with a
typed reason (`error_max_budget_usd`). We only count attempts.

Design: env `GEMINI_MAX_BUDGET_USD` / `GEMINI_MAX_TOTAL_TOKENS` (0 = off),
checked at the top of each attempt; result gains
`stop_reason in {accepted, attempts_exhausted, budget_exhausted, cancelled}`
and `usage_total`. Studio DATA panel shows the stop reason. Default behavior
unchanged.

Touches: ~15 lines in `service.py`, two lines in web types/metadata.

## C. Engineering discipline

### 6. JSONL typed progress events

Today `job_progress.py` rewrites one whole-JSON snapshot per checkpoint and
the UI only sees attempt-level state.

Design: append-only JSONL with a fixed event taxonomy (`attempt_start`,
`model_returned`, `exec_result`, `validator_verdict`, `attempt_end`,
`run_end{stop_reason}`), all keyed by `attempt`. The server's GET aggregates
into the existing response shape, so the frontend migrates for free; later
the Studio log can render sub-phases ("executing sandbox", "validating") and
every job leaves a complete replayable timeline.

Touches: `job_progress.py` + the read path in `server.py`.

### 7. Birth certificates on prompt rules

Claude Code annotates prompt rules with origin PRs, A/B experiment names,
measured effects ("~1.2% output-token reduction vs qualitative wording"), and
`@[MODEL LAUNCH]` removal reminders. Prompt rules are code with lifecycles.

Design: pure comment discipline, applied as we touch files. Example: the
orientation line in `REFERENCE_IMAGE_GUIDANCE` should carry
`# added 2026-07-07: fixes vertical-mirror transcription; retest before removing`.

## D. Architecture experiment

### 8. Mini-agent inside one attempt

The one real capability gap: our model repairs blind — it never sees its
program run; feedback arrives one full round-trip later. Claude Code's model
executes and inspects mid-turn, which is the root of its single-turn quality.

Design: keep the outer deterministic workflow (attempt cap, budget, progress
file) untouched; inside a single attempt, expose Gemini function-calling
tools — `execute_and_validate(code)` returning the #1-format delta, possibly
`inspect_region(x, y, w, h)` — capped at 2-3 tool calls, so the model can
self-check before submitting. This is the "agent inside a workflow" hybrid;
Claude Code itself embeds fixed workflows (structured-output retries) inside
its agent.

Gate: same benchmark as batch A; adopt only if acceptance/attempts/cost beat
the plain loop.

## Sequencing

1. Batch A: #1 + #2 + #3 (+#7 in passing) — one theme, one regression run.
2. Batch B: #4 + #5.
3. Batch C: #6.
4. Experiment #8 rides on batch A's benchmark.
