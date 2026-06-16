/* eslint-disable @typescript-eslint/no-require-imports */
// Starts both backend servers concurrently:
//   8000 — GSDiff Python backend  (root .venv)
//   8100 — hfagent FastAPI        (hfagent/.venv)
const { spawn } = require('child_process');
const path = require('path');

const root = path.join(__dirname, '..');

const servers = [
  {
    label: 'gsdiff',
    python: path.join(root, '.venv', 'Scripts', 'python.exe'),
    args: ['-m', 'uvicorn', 'app.main:app', '--port', '8000'],
    cwd: path.join(__dirname, 'backend'),
  },
  {
    label: 'hfagent',
    python: path.join(root, 'hfagent', '.venv', 'Scripts', 'python.exe'),
    args: ['-m', 'uvicorn', 'hfagent.api:app', '--port', '8100'],
    cwd: root,
  },
];

for (const { label, python, args, cwd } of servers) {
  const proc = spawn(python, args, { cwd, stdio: 'inherit', shell: false });
  proc.on('error', (err) => console.error(`[${label}] failed to start:`, err.message));
  proc.on('exit', (code) => { if (code !== 0) console.error(`[${label}] exited with code`, code); });
}
