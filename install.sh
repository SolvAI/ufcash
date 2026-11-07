#!/bin/sh
# UFCash installer for macOS and Linux:  curl -LsSf https://raw.githubusercontent.com/SolvAI/ufcash/main/install.sh | sh
# (or ./install.sh inside a clone). Installs uv if needed, Python 3.12 + dependencies in ./.venv, then downloads the data.
set -e
REPO="https://github.com/SolvAI/ufcash"
say() { printf '\033[36;1m>\033[0m %s\n' "$1"; }

if [ ! -f ufcash.py ]; then
  if [ -f "$(dirname "$0")/ufcash.py" ]; then
    cd "$(dirname "$0")"
  else
    say "Downloading UFCash into ./ufcash"
    if command -v git >/dev/null 2>&1; then
      git clone --depth 1 "$REPO.git" ufcash
    else
      curl -LsSf "$REPO/archive/refs/heads/main.tar.gz" | tar xz && mv ufcash-main ufcash
    fi
    cd ufcash
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
say "Installing Python 3.12 and the dependencies into ./.venv"
uv venv --quiet --allow-existing --python 3.12 .venv
uv pip install --quiet --python .venv/bin/python -r requirements.txt
chmod +x ufcash
./ufcash setup
printf '\n\033[1mInstalled.\033[0m  cd %s, edit data/upcoming.csv, then:  ./ufcash card\n' "$(pwd)"
