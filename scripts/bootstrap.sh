#!/usr/bin/env bash
set -euo pipefail

INSTALL_PROFILE="${INSTALL_PROFILE:-full}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Check Python version (Mine needs 3.11+)
check_python_version() {
  local py_version
  py_version=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
  local major minor
  major=$(echo "$py_version" | cut -d. -f1)
  minor=$(echo "$py_version" | cut -d. -f2)

  if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -lt 11 ]]; }; then
    echo "ERROR: Mine needs Python 3.11+, but found Python $py_version"
    echo ""
    echo "Fix:"
    echo "  # Windows: Download from https://python.org"
    echo "  # macOS:   brew install python@3.13"
    echo "  # Linux:   apt install python3.13  OR  pyenv install 3.13"
    echo ""
    echo "Then re-run with:"
    echo "  PYTHON_BIN=/path/to/python3.13 bash scripts/bootstrap.sh"
    exit 1
  fi
  echo "Python version: $py_version ✓"
}

# Check Node.js version (awp-wallet needs 20+)
check_node_version() {
  if ! command -v node >/dev/null 2>&1; then
    echo "WARNING: Node.js not found (awp-wallet requires Node.js 20+)"
    echo "  Install: https://nodejs.org or use nvm/fnm"
    return 0  # Continue, but warn
  fi

  local node_version
  node_version=$(node -v 2>/dev/null | sed 's/v//' || echo "0.0.0")
  local major
  major=$(echo "$node_version" | cut -d. -f1)

  if [[ "$major" -lt 20 ]]; then
    echo "WARNING: awp-wallet needs Node.js 20+, but found v$node_version"
    echo "  Upgrade: nvm install 20 && nvm use 20"
    return 0  # Continue, but warn
  fi
  echo "Node.js version: v$node_version ✓"
}

check_host_dependencies() {
  "$PYTHON_BIN" scripts/host_diagnostics.py --json >/tmp/mine-host-diagnostics.json || true
}

install_requirements() {
  echo "Installing core dependencies..."
  "$VENV_DIR/bin/python" -m pip install -r requirements-core.txt || {
    echo "WARNING: Some core dependencies failed to install. Retrying individually..."
    # pip install -r can fail entirely if ONE package fails to build
    # (e.g. crawl4ai, PyMuPDF on exotic platforms). Fall back to
    # installing each line individually so websockets/markdownify/etc.
    # still get installed even if a native-extension package fails.
    while IFS= read -r line; do
      line="${line%%#*}"           # strip comments
      line="$(echo "$line" | xargs)"  # trim whitespace
      [[ -z "$line" ]] && continue
      "$VENV_DIR/bin/python" -m pip install "$line" || \
        echo "WARNING: failed to install $line (continuing)"
    done < requirements-core.txt
  }
  if [[ "$INSTALL_PROFILE" == "browser" || "$INSTALL_PROFILE" == "full" ]]; then
    if [[ -f requirements-browser.txt ]]; then
      "$VENV_DIR/bin/python" -m pip install -r requirements-browser.txt || \
        echo "WARNING: Some browser dependencies failed to install."
    fi
    # Download Playwright browser binaries (chromium only to save space)
    echo "Installing Playwright browsers..."
    "$VENV_DIR/bin/python" -m playwright install chromium 2>/dev/null || \
      echo "WARNING: Playwright browser install failed. Amazon/LinkedIn crawling may not work."
  fi
  if [[ "$INSTALL_PROFILE" == "full" && -f requirements-dev.txt ]]; then
    "$VENV_DIR/bin/python" -m pip install -r requirements-dev.txt || \
      echo "WARNING: Dev dependencies failed to install (non-fatal)."
  fi
}

# Run version checks first
echo "Checking prerequisites..."
check_python_version
check_node_version
echo ""

if [[ -d "$VENV_DIR" ]]; then
  echo "reusing existing virtualenv: $VENV_DIR"
else
  # Try uv first, fall back to python -m venv
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed "$VENV_DIR"
  else
    echo "uv not found, using python -m venv (consider installing uv for faster installs)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

check_host_dependencies

# Install awp-wallet if not present
if ! command -v awp-wallet >/dev/null 2>&1; then
  echo "Installing awp-wallet from GitHub..."

  # Check prerequisites
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found. Please install git"
    exit 1
  fi

  if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: Node.js not found. Please install Node.js 20+ from https://nodejs.org"
    exit 1
  fi

  # Determine version to install (prefer latest tag, fallback to main)
  AWP_WALLET_VERSION="${AWP_WALLET_VERSION:-}"
  if [ -z "$AWP_WALLET_VERSION" ]; then
    AWP_WALLET_VERSION=$(git ls-remote --tags --sort=-v:refname https://github.com/awp-core/awp-wallet.git 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  fi
  AWP_WALLET_VERSION="${AWP_WALLET_VERSION:-main}"
  echo "  Target version: $AWP_WALLET_VERSION"

  # Clone and install from GitHub
  TEMP_DIR=$(mktemp -d)
  trap "rm -rf $TEMP_DIR" EXIT

  git clone --branch "$AWP_WALLET_VERSION" --depth 1 https://github.com/awp-core/awp-wallet.git "$TEMP_DIR"
  cd "$TEMP_DIR"
  npm install
  npm install -g .
  cd -

  echo "awp-wallet $AWP_WALLET_VERSION installed successfully from GitHub ✓"
else
  echo "awp-wallet already installed: $(command -v awp-wallet) ✓"
fi
echo ""

install_requirements
"$VENV_DIR/bin/python" scripts/verify_env.py --profile "$INSTALL_PROFILE"
"$VENV_DIR/bin/python" scripts/smoke_test.py

echo ""
echo "Running post-install check..."
"$VENV_DIR/bin/python" scripts/post_install_check.py
