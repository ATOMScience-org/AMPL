#!/bin/bash
# No fancy cd logic needed if you run it from the root
# but let's keep it robust just in case.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Sync everything defined in pyproject.toml
# This creates/updates the .venv in the root
uv sync --all-extras

# Now this actually works because '.' contains the 'atomsci' folder
# and the pyproject.toml instructions to find it.
uv pip install -e .

echo "AMPL environment synced and atomsci.ddm linked in editable mode."
