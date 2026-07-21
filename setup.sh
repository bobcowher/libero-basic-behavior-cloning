#!/usr/bin/env bash
#
# setup.sh — runs on the training server BEFORE `pip install -r requirements.txt`.
# Handles the LIBERO setup that requirements.txt CANNOT express:
#   1. system libs for headless EGL/mujoco rendering        (best-effort)
#   2. the LIBERO source itself (git clone)                 (not on PyPI)
#   3. editable install of LIBERO in *compat* mode          (critical)
#   4. robosuite macros                                     (best-effort, cosmetic)
#
# Ordering note: this script runs BEFORE requirements.txt, so robosuite/mujoco
# are NOT installed yet when it runs. Steps that need them (#4) are guarded and
# skip cleanly rather than fail. Step #3 is safe here because LIBERO's setup.py
# has an empty install_requires — the editable install pulls no dependencies.

set -euo pipefail

# ---- config (override via env) ----
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"
# Pin a known-good revision for reproducibility by exporting LIBERO_COMMIT=<sha>.
LIBERO_COMMIT="${LIBERO_COMMIT:-}"
PY="${PYTHON:-python}"

echo "[setup] LIBERO_DIR=$LIBERO_DIR"

# ---- 1. system libs for headless mujoco/EGL (best-effort) ----
# GPU training images usually already have these; don't fail the run if apt is
# unavailable or we lack sudo.
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
  $SUDO apt-get update -y || true
  $SUDO apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \
    libglfw3 libegl1 patchelf \
    || echo "[setup] apt step skipped/failed — continuing"
fi

# ---- 2. LIBERO source ----
if [ ! -d "$LIBERO_DIR/.git" ]; then
  echo "[setup] cloning LIBERO -> $LIBERO_DIR"
  git clone "$LIBERO_REPO" "$LIBERO_DIR"
else
  echo "[setup] LIBERO already present at $LIBERO_DIR"
fi
if [ -n "$LIBERO_COMMIT" ]; then
  echo "[setup] checking out LIBERO@$LIBERO_COMMIT"
  git -C "$LIBERO_DIR" fetch --all --quiet || true
  git -C "$LIBERO_DIR" checkout "$LIBERO_COMMIT"
fi

# ---- 3. editable install of LIBERO (compat mode — CRITICAL) ----
# Plain `pip install -e` installs an EMPTY package here: LIBERO/libero/ has no
# __init__.py (implicit namespace dir), so classic find_packages() returns
# nothing and the PEP 660 strict finder exposes zero modules -> `import libero`
# fails. compat mode uses the legacy flat .pth-on-repo-root layout, which
# imports the namespace package correctly.
echo "[setup] editable-installing LIBERO (compat mode)"
$PY -m pip install -e "$LIBERO_DIR" --config-settings editable_mode=compat

# ---- 4. robosuite macros (best-effort, cosmetic) ----
# robosuite arrives with requirements.txt AFTER this script, so it's usually not
# importable yet. Skip cleanly if so — robosuite runs fine without the private
# macros file (it just prints a startup notice). If you want it applied, run the
# printed command once after `pip install -r requirements.txt`.
if $PY -c "import robosuite" >/dev/null 2>&1; then
  RS_DIR="$($PY -c "import robosuite, os; print(os.path.dirname(robosuite.__file__))")"
  echo "[setup] running robosuite setup_macros.py"
  $PY "$RS_DIR/scripts/setup_macros.py" || true
else
  echo "[setup] robosuite not installed yet — skipping setup_macros (optional)."
  echo "        Run once AFTER requirements install if you want it:"
  echo "        \$PY \$(\$PY -c 'import robosuite,os;print(os.path.dirname(robosuite.__file__))')/scripts/setup_macros.py"
fi

echo "[setup] done. requirements.txt will install the rest (torch cu128, robosuite==1.4.0, mujoco==2.3.2, ...)."
