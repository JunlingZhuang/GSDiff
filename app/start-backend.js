/* eslint-disable @typescript-eslint/no-require-imports */
const { execSync } = require('child_process');
const path = require('path');

const backendDir = path.join(__dirname, 'backend');
const python = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');

execSync(`"${python}" -m uvicorn app.main:app --port 8000`, {
  cwd: backendDir,
  stdio: 'inherit',
});
