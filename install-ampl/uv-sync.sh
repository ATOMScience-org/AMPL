DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

PATH=~/.local/bin:$PATH
UV_CMD=$(which uv)

if [ "$UV_CMD" = "" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

. ampl-venv.sh || exit 1

# NOTE add --refresh to get latest version of repos
uv sync --active

