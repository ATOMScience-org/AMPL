#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

VENV_ID=ampl
SYNC_PACKAGES=0

PATH=~/.local/bin:$PATH
UV_CMD=$(which uv)

if [ "$UV_CMD" = "" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

OPTIONS=(
  "python/3.10" "Python 3.10 " off
  "python/3.11" "Python 3.11 [deepchem support questionable]" off
)

# Show checklist and capture selected values
CHOICES=$(dialog --clear --title "Select Options" \
  --checklist "Use SPACE to select/unselect, ENTER to confirm:" 20 60 10 \
  "${OPTIONS[@]}" \
  3>&1 1>&2 2>&3)

dialog --clear \
  --title "Sync Packages" \
  --yesno "Do you want to sync packages in the venvs?" 8 60

response=$?

if [[ $response -eq 0 ]]; then
    SYNC_PACKAGES=1
else
    SYNC_PACKAGES=0
fi

clear  # remove dialog UI

# CHOICES is a quoted space-separated list of values, e.g. "build" "limits"
# Remove quotes and loop
for choice in $CHOICES; do
    # Remove surrounding quotes
    val="${choice%\"}"
    val="${val#\"}"
    module_name=$val
    python_version="${val#*/}"
    python_version_str="${python_version//./}"
    venv_name="${VENV_ID}-py${python_version_str}-venv"
    module load $module_name
    if [ ! -e $(pwd)/venvs/$venv_name ]; then
      uv venv --python $(which python3) $(pwd)/venvs/$venv_name
    fi
    if [ $SYNC_PACKAGES -eq 1 ]; then
      ./uv-sync.sh $python_version
    fi
done

