#!/bin/bash

# Script to generate both CPU and GPU requirements files for bayalign
# Usage: ./export_requirements.sh

# Set error handling
set -e

echo "Generating CPU requirements file..."
uv export --extra all --no-emit-project --no-hashes -o requirements-cpu.txt
echo "✅ CPU requirements exported to requirements-cpu.txt"

echo "Generating GPU requirements file..."
uv export --extra all-gpu --no-emit-project --no-hashes -o requirements-gpu.txt
echo "✅ GPU requirements exported to requirements-gpu.txt"

echo "Done! Generated both requirements files successfully."
echo "  - requirements-cpu.txt: CPU version with all dependencies"
echo "  - requirements-gpu.txt: GPU version with all dependencies"