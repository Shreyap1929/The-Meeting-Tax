#!/usr/bin/env bash
set -euo pipefail  # Stop on errors, unset vars, and failed pipes.

# Safety check: warn if you're not inside the intended workspace folder.
if [ "$(basename "$PWD")" != "The Meeting Tax" ]; then
  echo "Warning: current folder is '$(basename "$PWD")', expected 'The Meeting Tax'."
fi

# Create the required directory tree in the current folder.
mkdir -p data/raw data/processed notebooks sql app images

# Create root files if they do not exist yet.
touch requirements.txt README.md .gitignore

echo "Scaffold created in: $PWD"