#!/bin/bash

# Remove all __pycache__, .pytest_cache, and .vscode directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null

# Find all Python files
PYPATH=$(find . -type f -name "*.py")

# Run Black to format the code
python3 -m black $PYPATH

# Run isort to sort imports
python3 -m isort $PYPATH
