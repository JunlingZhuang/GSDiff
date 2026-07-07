"use strict";
function element(id) {
    const found = document.getElementById(id);
    if (!found)
        throw new Error(`Missing element: ${id}`);
    return found;
}
const ui = {
    sample: element("sampleSelect"),
    editor: element("programEditor"),
    prompt: element("promptInput"),
    width: element("gridWidth"),
    height: element("gridHeight"),
    generate: element("generateButton"),
    status: element("requestStatus"),
    provider: element("providerStatus"),
    jsonStatus: element("jsonStatus"),
    canvas: element("planCanvas"),
    empty: element("canvasEmpty"),
    tooltip: element("canvasTooltip"),
    title: element("outputTitle"),
    subtitle: element("outputSubtitle"),
    score: element("scoreValue"),
    roomCount: element("roomCount"),
    legend: element("legendList"),
    validationCount: element("validationCount"),
    validation: element("validationList"),
    doorCount: element("doorCount"),
    doors: element("doorList"),
    code: element("codeOutput"),
    source: element("codeSource"),
    metadata: element("metadataGrid"),
    areas: element("areaTable"),
    png: element("downloadPng"),
    json: element("downloadJson"),
    copy: element("copyCode"),
    gridWidthLabel: element("gridWidthLabel"),
    generateLabel: element("generateLabel"),
    generateHint: element("generateHint"),
    generationOverlay: element("generationOverlay"),
    generationPhase: element("generationPhase"),
    generationElapsed: element("generationElapsed"),
    stopGeneration: element("stopGeneration"),
    iterationList: element("iterationList"),
    runCode: element("runCode"),
    inspectCode: element("inspectCode"),
    fixCode: element("fixCode"),
    reviseCode: element("reviseCode"),
    revision: element("revisionInput"),
    inspection: element("inspectionSummary"),
};
const state = {
    samples: {},
    result: null,
    activeController: null,
    activeRequestId: null,
    generationTimer: null,
    generationStartedAt: 0,
    previewPlan: null,
};
function prettyType(value) {
    return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}
function escapeHtml(value) {
    const replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;",
    };
    return String(value).replace(/[&<>'"]/g, (character) => replacements[character] ?? character);
}
async function initialize() {
    const [samplesResponse, healthResponse] = await Promise.all([
        fetch("/api/samples"),
        fetch("/api/health"),
    ]);
    if (!samplesResponse.ok || !healthResponse.ok)
        throw new Error("The backend did not initialize correctly.");
    state.samples = (await samplesResponse.json());
    const health = (await healthResponse.json());
    ui.provider.classList.toggle("ready", health.gemini_configured);
    const modelRoute = health.fast_model && health.quality_model
        ? `${health.fast_model} → ${health.quality_model}${health.complex_model ? ` · complex ${health.complex_model}` : ""}`
        : health.model;
    ui.provider.innerHTML = `<b></b>${health.gemini_configured ? `Gemini ready · ${escapeHtml(modelRoute)}` : "Gemini API key required"}`;
    ui.sample.innerHTML = Object.entries(state.samples)
        .map(([key, sample]) => `<option value="${escapeHtml(key)}">${escapeHtml(key)} · ${escapeHtml(sample.building_type)}</option>`)
        .join("");
    loadSample("clinic-small");
    ui.status.textContent = "Ready · click Generate Plan";
}
function loadSample(key) {
    const sample = state.samples[key];
    if (!sample)
        return;
    ui.editor.value = JSON.stringify(sample, null, 2);
    ui.prompt.value = key === "inpatient-ward"
        ? "Place patient rooms along the perimeter, nurse stations centrally, and provide direct doors from every occupied room to circulation."
        : "Create a compact, legible plan. Put the main entrance on the west side and connect every occupied room to circulation with a door.";
    validateEditor();
}
function validateEditor() {
    try {
        const value = JSON.parse(ui.editor.value);
        const count = value.rooms?.reduce((total, room) => total + Number(room.count || 1), 0) ?? 0;
        ui.jsonStatus.textContent = `Valid JSON · ${count} requested spaces`;
        ui.jsonStatus.classList.remove("invalid");
        return value;
    }
    catch (error) {
        ui.jsonStatus.textContent = `Invalid JSON · ${error instanceof Error ? error.message : String(error)}`;
        ui.jsonStatus.classList.add("invalid");
        return null;
    }
}
function setAgentButtons(enabled) {
    const hasCode = Boolean(state.result && ui.code.value.trim());
    [ui.runCode, ui.inspectCode, ui.fixCode].forEach((button) => {
        button.disabled = !enabled || !hasCode;
    });
    ui.reviseCode.disabled = !enabled || !hasCode;
}
function actionCopy(action) {
    const values = {
        generate: { phase: "Gemini is writing a complete Python program", status: "Generating, running, and validating Python..." },
        run: { phase: "Executing the edited Python program", status: "Running the current code in the isolated worker..." },
        inspect: { phase: "Inspecting geometry and architectural constraints", status: "Executing and inspecting the current code..." },
        fix: { phase: "Gemini is diagnosing and fixing the current code", status: "Running the AI repair loop, up to five attempts..." },
        revise: { phase: "Gemini is revising the complete Python program", status: "Applying the revision, then executing and validating it..." },
    };
    return values[action];
}
function renderGenerationProgress(progress, program) {
    renderIterations(progress.iterations);
    const latest = progress.iterations[progress.iterations.length - 1];
    if (latest) {
        const score = latest.status === "running"
            ? "waiting for Gemini"
            : latest.score === null ? "no executable plan" : `score ${latest.score}`;
        ui.generationPhase.textContent = `Attempt ${latest.attempt} ${latest.phase} · ${latest.status} · ${score}`;
        ui.status.textContent = latest.status === "running"
            ? `Live · attempt ${latest.attempt} is waiting for ${latest.model ?? "Gemini"}`
            : `Live checkpoint · attempt ${latest.attempt} ${latest.status} · continuing agent loop`;
    }
    else if (progress.message) {
        ui.generationPhase.textContent = progress.message;
    }
    const preview = progress.preview;
    if (!preview)
        return;
    if (preview.code) {
        ui.code.value = preview.code;
        const identity = [preview.source ?? "gemini", preview.model].filter(Boolean).join(" · ");
        ui.source.textContent = `LIVE · ${identity}`;
    }
    if (!preview.plan || !preview.validation)
        return;
    state.previewPlan = preview.plan;
    ui.empty.hidden = true;
    ui.title.textContent = program.building_type;
    ui.subtitle.textContent = `${preview.plan.width} × ${preview.plan.height} pixels · live attempt checkpoint`;
    ui.score.textContent = String(preview.validation.score);
    ui.gridWidthLabel.textContent = `${preview.plan.width} PX`;
    renderCanvas(preview.plan);
    renderLegend(preview.plan);
    renderValidation(preview.validation);
    renderDoors(preview.plan.doors);
}
async function pollGenerationProgress(requestId, program, signal, isFinished) {
    while (!signal.aborted && !isFinished()) {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        if (signal.aborted || isFinished())
            return;
        try {
            const response = await fetch(`/api/generate/${encodeURIComponent(requestId)}`, {
                method: "GET",
                signal,
                cache: "no-store",
            });
            if (!response.ok)
                continue;
            renderGenerationProgress((await response.json()), program);
        }
        catch (error) {
            if (!(error instanceof Error) || error.name !== "AbortError")
                return;
        }
    }
}
async function requestAgentAction(action) {
    if (state.activeRequestId)
        return;
    const program = validateEditor();
    if (!program)
        return;
    if (action !== "generate" && !ui.code.value.trim()) {
        ui.status.className = "request-status error";
        ui.status.textContent = "Generate a plan before using code-agent actions.";
        return;
    }
    if (action === "revise" && !ui.revision.value.trim()) {
        ui.status.className = "request-status error";
        ui.status.textContent = "Enter a revision instruction first.";
        return;
    }
    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    const copy = actionCopy(action);
    state.activeRequestId = requestId;
    state.activeController = controller;
    state.generationStartedAt = performance.now();
    ui.generate.disabled = true;
    setAgentButtons(false);
    ui.generate.classList.add("is-loading");
    ui.generate.setAttribute("aria-busy", "true");
    ui.generateLabel.textContent = action === "generate" ? "GENERATING PLAN" : action.toUpperCase();
    ui.generateHint.textContent = "use Stop Generation to cancel";
    ui.stopGeneration.disabled = false;
    ui.generationOverlay.hidden = false;
    ui.empty.hidden = true;
    ui.generationElapsed.textContent = "0s";
    ui.generationPhase.textContent = copy.phase;
    state.generationTimer = window.setInterval(() => {
        const elapsed = Math.floor((performance.now() - state.generationStartedAt) / 1000);
        ui.generationElapsed.textContent = `${elapsed}s`;
        if (elapsed >= 12 && action !== "run" && action !== "inspect") {
            ui.generationPhase.textContent = "Executing, inspecting, and repairing the current candidate";
        }
    }, 250);
    ui.status.className = "request-status";
    ui.status.textContent = copy.status;
    const started = performance.now();
    const endpoint = action === "run" || action === "inspect" ? "/api/execute" : "/api/generate";
    let requestFinished = false;
    const progressTask = endpoint === "/api/generate"
        ? pollGenerationProgress(requestId, program, controller.signal, () => requestFinished)
        : Promise.resolve();
    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
                request_id: requestId,
                action,
                program,
                prompt: action === "revise" ? ui.revision.value : ui.prompt.value,
                code: action === "run" || action === "inspect" ? ui.code.value : undefined,
                current_code: action === "fix" || action === "revise" ? ui.code.value : undefined,
                mode: "auto",
                options: { width: Number(ui.width.value), height: Number(ui.height.value) },
            }),
        });
        const payload = (await response.json());
        if (!response.ok || "error" in payload)
            throw new Error("error" in payload ? payload.error : "Generation failed.");
        state.result = payload;
        renderResult(payload);
        if (action === "inspect")
            switchView("data");
        const acceptance = payload.accepted === false ? " · needs revision" : "";
        ui.status.textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)}s · ${payload.source}${acceptance}`;
    }
    catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
            ui.status.className = "request-status";
            ui.status.textContent = "Generation stopped · adjust inputs or re-generate";
        }
        else {
            ui.status.className = "request-status error";
            ui.status.textContent = error instanceof Error ? error.message : String(error);
        }
    }
    finally {
        requestFinished = true;
        await progressTask;
        if (state.activeRequestId === requestId) {
            if (state.generationTimer !== null)
                window.clearInterval(state.generationTimer);
            state.generationTimer = null;
            state.activeController = null;
            state.activeRequestId = null;
            ui.generationOverlay.hidden = true;
            ui.generate.disabled = false;
            ui.generate.classList.remove("is-loading");
            ui.generate.setAttribute("aria-busy", "false");
            ui.stopGeneration.disabled = false;
            setAgentButtons(true);
            ui.generateLabel.textContent = state.result ? "RE-GENERATE PLAN" : "GENERATE PLAN";
            ui.generateHint.textContent = state.result ? "create another candidate" : "write code · run · inspect · repair";
            if (!state.result && !state.previewPlan)
                ui.empty.hidden = false;
        }
    }
}
async function generate() {
    await requestAgentAction("generate");
}
async function stopGeneration() {
    const requestId = state.activeRequestId;
    const controller = state.activeController;
    if (!requestId || !controller)
        return;
    ui.stopGeneration.disabled = true;
    ui.generationPhase.textContent = "Stopping the active generation job";
    ui.status.textContent = "Stopping generation...";
    try {
        await fetch(`/api/generate/${encodeURIComponent(requestId)}`, { method: "DELETE" });
    }
    catch {
        // The browser request is still aborted even if the cancellation acknowledgement fails.
    }
    finally {
        controller.abort();
    }
}
function renderResult(result) {
    state.previewPlan = result.plan;
    ui.empty.hidden = true;
    ui.title.textContent = result.program.building_type;
    ui.subtitle.textContent = `${result.plan.width} × ${result.plan.height} pixels · ${(result.plan.meters_per_cell * 3.28084).toFixed(2)} ft/cell · ${result.source}`;
    ui.score.textContent = String(result.validation.score);
    ui.gridWidthLabel.textContent = `${result.plan.width} PX`;
    ui.code.value = result.code;
    ui.source.textContent = result.model ? `${result.source} · ${result.model}` : result.source;
    [ui.png, ui.json, ui.copy].forEach((button) => { button.disabled = false; });
    setAgentButtons(true);
    if (result.inspection) {
        ui.inspection.className = `inspection-summary ${result.inspection.accepted ? "accepted" : "rejected"}`;
        ui.inspection.textContent = result.inspection.summary;
    }
    else if (result.accepted === false) {
        ui.inspection.className = "inspection-summary rejected";
        ui.inspection.textContent = result.ai_error ?? "The current program executed but did not pass acceptance checks.";
    }
    else {
        ui.inspection.className = "inspection-summary accepted";
        ui.inspection.textContent = "The active program executed and passed the current acceptance checks.";
    }
    renderCanvas(result.plan);
    renderLegend(result.plan);
    renderValidation(result.validation);
    renderDoors(result.plan.doors);
    renderData(result);
    renderIterations(result.iterations);
}
function renderCanvas(plan) {
    const scale = Math.max(6, Math.min(14, Math.floor(1280 / plan.width)));
    const deviceScale = Math.min(2, window.devicePixelRatio || 1);
    ui.canvas.width = plan.width * scale * deviceScale;
    ui.canvas.height = plan.height * scale * deviceScale;
    ui.canvas.style.width = `${plan.width * scale}px`;
    ui.canvas.style.height = `${plan.height * scale}px`;
    const context = ui.canvas.getContext("2d");
    if (!context)
        throw new Error("Canvas rendering is unavailable.");
    context.scale(deviceScale, deviceScale);
    context.fillStyle = "#d8d3c8";
    context.fillRect(0, 0, plan.width * scale, plan.height * scale);
    for (let y = 0; y < plan.height; y += 1) {
        for (let x = 0; x < plan.width; x += 1) {
            const offset = y * plan.width + x;
            if (plan.footprint?.[offset]) {
                context.fillStyle = "#f9f7f0";
                context.fillRect(x * scale, y * scale, scale + 0.4, scale + 0.4);
            }
        }
    }
    for (let y = 0; y < plan.height; y += 1) {
        for (let x = 0; x < plan.width; x += 1) {
            const roomIndex = plan.cells[y * plan.width + x];
            const room = roomIndex === undefined || roomIndex < 0 ? undefined : plan.rooms[roomIndex];
            if (!room)
                continue;
            context.fillStyle = room.color;
            context.fillRect(x * scale, y * scale, scale + 0.4, scale + 0.4);
        }
    }
    context.strokeStyle = "rgba(24,31,29,.72)";
    context.lineWidth = Math.max(1, scale * 0.09);
    context.beginPath();
    for (let y = 0; y < plan.height; y += 1) {
        for (let x = 0; x < plan.width; x += 1) {
            const index = plan.cells[y * plan.width + x];
            if (index === undefined || index < 0)
                continue;
            if (x === 0 || plan.cells[y * plan.width + x - 1] !== index) {
                context.moveTo(x * scale, y * scale);
                context.lineTo(x * scale, (y + 1) * scale);
            }
            if (y === 0 || plan.cells[(y - 1) * plan.width + x] !== index) {
                context.moveTo(x * scale, y * scale);
                context.lineTo((x + 1) * scale, y * scale);
            }
            if (x === plan.width - 1) {
                context.moveTo((x + 1) * scale, y * scale);
                context.lineTo((x + 1) * scale, (y + 1) * scale);
            }
            if (y === plan.height - 1) {
                context.moveTo(x * scale, (y + 1) * scale);
                context.lineTo((x + 1) * scale, (y + 1) * scale);
            }
        }
    }
    context.stroke();
    if (scale >= 7 && plan.rooms.length <= 90) {
        context.textAlign = "center";
        context.textBaseline = "middle";
        for (const [roomIndex, room] of plan.rooms.entries()) {
            const bounds = room.bounds;
            if (bounds.width * scale < 32 || bounds.height * scale < 20)
                continue;
            const targetX = bounds.x + bounds.width / 2;
            const targetY = bounds.y + bounds.height / 2;
            let labelX = targetX;
            let labelY = targetY;
            let bestDistance = Number.POSITIVE_INFINITY;
            for (let y = bounds.y; y < bounds.y + bounds.height; y += 1) {
                for (let x = bounds.x; x < bounds.x + bounds.width; x += 1) {
                    if (plan.cells[y * plan.width + x] !== roomIndex)
                        continue;
                    const distance = (x + 0.5 - targetX) ** 2 + (y + 0.5 - targetY) ** 2;
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        labelX = x + 0.5;
                        labelY = y + 0.5;
                    }
                }
            }
            const label = room.id.replaceAll("_", " ");
            context.font = `700 ${Math.max(7, Math.min(10, scale * 0.72))}px ui-monospace, monospace`;
            context.fillStyle = "rgba(18,25,23,.8)";
            context.fillText(label.length > 17 ? `${label.slice(0, 15)}...` : label, labelX * scale, labelY * scale);
        }
    }
    plan.doors.forEach((door) => drawDoor(context, door, scale));
}
function drawDoor(context, door, scale) {
    const width = door.width_cells * scale;
    const x = door.x * scale;
    const y = door.y * scale;
    context.save();
    context.lineCap = "square";
    context.strokeStyle = "#fffdf8";
    context.lineWidth = Math.max(3, scale * 0.35);
    context.beginPath();
    if (door.orientation === "horizontal") {
        context.moveTo(x, y);
        context.lineTo(x + width, y);
    }
    else {
        context.moveTo(x, y);
        context.lineTo(x, y + width);
    }
    context.stroke();
    context.strokeStyle = door.to_room === null ? "#f0643b" : "#17211f";
    context.lineWidth = Math.max(1, scale * 0.12);
    context.beginPath();
    let openAngle = 0;
    let closedAngle = 0;
    if (door.orientation === "horizontal") {
        openAngle = door.swing_side === "south" ? Math.PI / 2 : -Math.PI / 2;
        closedAngle = 0;
        context.moveTo(x, y);
        context.lineTo(x + Math.cos(openAngle) * width, y + Math.sin(openAngle) * width);
        context.moveTo(x + width, y);
        context.arc(x, y, width, closedAngle, openAngle, openAngle < closedAngle);
    }
    else {
        openAngle = door.swing_side === "west" ? Math.PI : 0;
        closedAngle = Math.PI / 2;
        context.moveTo(x, y);
        context.lineTo(x + Math.cos(openAngle) * width, y + Math.sin(openAngle) * width);
        context.moveTo(x, y + width);
        context.arc(x, y, width, closedAngle, openAngle, openAngle < closedAngle);
    }
    context.stroke();
    context.restore();
}
function renderLegend(plan) {
    const groups = new Map();
    for (const room of plan.rooms) {
        const item = groups.get(room.type) ?? { count: 0, color: room.color };
        item.count += 1;
        groups.set(room.type, item);
    }
    ui.roomCount.textContent = `${plan.rooms.length} SPACES`;
    ui.legend.innerHTML = [...groups.entries()]
        .map(([type, item]) => `<div class="legend-item"><i class="legend-color" style="background:${item.color}"></i><b>${escapeHtml(prettyType(type))}</b><span>×${item.count}</span></div>`)
        .join("");
}
function renderValidation(validation) {
    const passed = validation.checks.filter((check) => check.pass).length;
    ui.validationCount.textContent = `${passed}/${validation.checks.length} PASS`;
    ui.validation.innerHTML = validation.checks.slice(0, 30)
        .map((check) => `<div class="check-item ${check.pass ? "" : "fail"}"><span class="check-icon">${check.pass ? "✓" : "!"}</span><span>${escapeHtml(check.label)}</span></div>`)
        .join("");
}
function renderDoors(doors) {
    ui.doorCount.textContent = String(doors.length);
    ui.doors.innerHTML = doors.slice(0, 28)
        .map((door) => `<div class="door-item"><b>${escapeHtml(door.id)}</b><span>${escapeHtml(door.from_room)} → ${escapeHtml(door.to_room ?? "OUTSIDE")} · swings ${door.swing_side} @ ${door.x},${door.y}</span></div>`)
        .join("") || '<p class="empty-copy">No doors generated.</p>';
}
function renderData(result) {
    const items = [
        ["Source", result.source],
        ["Model", result.model ?? "None"],
        ["Score", `${result.validation.score}/100`],
        ["Grid", `${result.plan.width} × ${result.plan.height}`],
        ["Scale", `${result.plan.meters_per_cell.toFixed(3)} m/cell`],
        ["Rooms", result.plan.rooms.length],
        ["Doors", result.plan.doors.length],
        ["Coverage", `${result.validation.summary.coverage_percent}%`],
        ["Rules", result.validation.summary.rule_profile ?? "generic-schematic"],
    ];
    ui.metadata.innerHTML = items
        .map(([label, value]) => `<div class="metadata-item"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`)
        .join("");
    ui.areas.innerHTML = result.validation.areas
        .map((row) => `<tr><td>${escapeHtml(row.id)}</td><td>${escapeHtml(prettyType(row.type))}</td><td>${row.target_ft2 ?? Math.round(row.target_m2 * 10.7639)} sf</td><td>${row.actual_ft2 ?? Math.round(row.actual_m2 * 10.7639)} sf</td><td class="${row.pass ? "area-good" : "area-bad"}">${row.error_percent}%</td></tr>`)
        .join("") || '<tr><td colspan="5">No area data.</td></tr>';
}
function renderIterations(iterations) {
    ui.iterationList.innerHTML = iterations.map((iteration) => {
        const issues = iteration.issues.slice(0, 4)
            .map((issue) => `<li>${escapeHtml(issue)}</li>`)
            .join("");
        const durationMs = iteration.status === "running" && iteration.started_at_ms
            ? Math.max(0, Date.now() - iteration.started_at_ms)
            : iteration.duration_ms;
        const usage = iteration.usage;
        const tokenCount = usage ? usage.total_tokens.toLocaleString() : "—";
        const cost = usage?.estimated_cost_usd == null ? "—" : `$${usage.estimated_cost_usd.toFixed(3)}`;
        return `<article class="iteration-item ${escapeHtml(iteration.status)}">
      <div class="iteration-number">ATTEMPT<b>${String(iteration.attempt).padStart(2, "0")}</b></div>
      <div class="iteration-body">
        <h4>${escapeHtml(iteration.phase)} · ${escapeHtml(iteration.model ?? iteration.source)} · ${escapeHtml(iteration.status)}</h4>
        <p>${escapeHtml(iteration.message)}</p>
        ${issues ? `<ul class="iteration-issues">${issues}</ul>` : ""}
        ${iteration.code ? `<details class="iteration-code"><summary>VIEW ATTEMPT CODE</summary><pre><code>${escapeHtml(iteration.code)}</code></pre></details>` : ""}
      </div>
      <div class="iteration-metrics">
        <span>SCORE<b>${iteration.score ?? "—"}</b></span>
        <span>TIME<b>${(durationMs / 1000).toFixed(1)}s</b></span>
        <span>TOKENS<b>${tokenCount}</b></span>
        <span>EST. COST<b>${cost}</b></span>
      </div>
    </article>`;
    }).join("") || '<p class="empty-copy">No attempts were recorded.</p>';
}
function switchView(view) {
    document.querySelectorAll(".view-tab").forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.view === view);
    });
    document.querySelectorAll(".view-panel").forEach((panel) => panel.classList.remove("active"));
    const target = document.getElementById(`${view}View`);
    if (target)
        target.classList.add("active");
}
function download(name, type, content) {
    const blob = content instanceof Blob ? content : new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
ui.sample.addEventListener("change", () => loadSample(ui.sample.value));
ui.editor.addEventListener("input", validateEditor);
element("formatJson").addEventListener("click", () => {
    const value = validateEditor();
    if (value)
        ui.editor.value = JSON.stringify(value, null, 2);
});
ui.generate.addEventListener("click", () => { void generate(); });
ui.stopGeneration.addEventListener("click", () => { void stopGeneration(); });
ui.runCode.addEventListener("click", () => { void requestAgentAction("run"); });
ui.inspectCode.addEventListener("click", () => { void requestAgentAction("inspect"); });
ui.fixCode.addEventListener("click", () => { void requestAgentAction("fix"); });
ui.reviseCode.addEventListener("click", () => { void requestAgentAction("revise"); });
document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view ?? "plan"));
});
ui.png.addEventListener("click", () => {
    ui.canvas.toBlob((blob) => { if (blob)
        download("pixel-plan.png", "image/png", blob); }, "image/png");
});
ui.json.addEventListener("click", () => {
    if (state.result)
        download("pixel-plan.json", "application/json", JSON.stringify(state.result, null, 2));
});
ui.copy.addEventListener("click", () => {
    if (!state.result)
        return;
    void navigator.clipboard.writeText(ui.code.value).then(() => {
        ui.copy.textContent = "COPIED";
        window.setTimeout(() => { ui.copy.textContent = "COPY CODE"; }, 1200);
    });
});
ui.canvas.addEventListener("mousemove", (event) => {
    const plan = state.previewPlan;
    if (!plan)
        return;
    const bounds = ui.canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - bounds.left) / bounds.width) * plan.width);
    const y = Math.floor(((event.clientY - bounds.top) / bounds.height) * plan.height);
    const roomIndex = plan.cells[y * plan.width + x];
    const room = roomIndex === undefined || roomIndex < 0 ? undefined : plan.rooms[roomIndex];
    if (!room) {
        ui.tooltip.hidden = true;
        return;
    }
    const area = room.pixel_count * plan.meters_per_cell ** 2;
    ui.tooltip.innerHTML = `<b>${escapeHtml(room.id)}</b><br>${escapeHtml(prettyType(room.type))} · ${(area * 10.7639).toFixed(0)} sf<br>pixel (${x}, ${y})`;
    const stage = ui.canvas.closest(".canvas-stage")?.getBoundingClientRect();
    if (stage) {
        ui.tooltip.style.left = `${event.clientX - stage.left + 13}px`;
        ui.tooltip.style.top = `${event.clientY - stage.top + 13}px`;
    }
    ui.tooltip.hidden = false;
});
ui.canvas.addEventListener("mouseleave", () => { ui.tooltip.hidden = true; });
void initialize().catch((error) => {
    ui.status.className = "request-status error";
    ui.status.textContent = error instanceof Error ? error.message : String(error);
});
