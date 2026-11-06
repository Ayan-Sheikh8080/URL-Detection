// npm start uses the project's Python environment to run the FastAPI service.
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const localPython = process.platform === 'win32'
  ? path.join(root, '.venv', 'Scripts', 'python.exe')
  : path.join(root, '.venv', 'bin', 'python');
const python = process.env.PYTHON
  || (fs.existsSync(localPython) ? localPython : (process.platform === 'win32' ? 'python' : 'python3'));
const child = spawn(python, ['-m', 'src.start'], {
  cwd: root,
  env: process.env,
  stdio: 'inherit',
});

child.on('error', (error) => {
  console.error(`Unable to start Python API: ${error.message}`);
  process.exitCode = 1;
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.exitCode = 1;
  } else {
    process.exitCode = code === null ? 1 : code;
  }
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    if (!child.killed) child.kill(signal);
  });
}
