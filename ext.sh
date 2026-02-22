#!/usr/bin/env bash
# Installs the Jupyter extension for Cursor server (remote/SSH mode).
# Needed because the GUI installer downloads a corrupted file, and the latest
# marketplace version may require a newer VS Code engine than Cursor ships with.

set -e

EXTENSION_VERSION="${1:-2025.4.1}"
VSIX_URL="https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-toolsai/vsextensions/jupyter/${EXTENSION_VERSION}/vspackage?targetPlatform=linux-x64"
VSIX_TMP="/tmp/ms-toolsai.jupyter-${EXTENSION_VERSION}.vsix"
EXTENSIONS_DIR="/root/.cursor-server/extensions"

# Find the cursor-server binary (the path contains a hash that changes with updates)
CURSOR_SERVER=$(find /root/.cursor-server/bin/linux-x64 -name "cursor-server" -path "*/bin/cursor-server" 2>/dev/null | head -1)
if [[ -z "$CURSOR_SERVER" ]]; then
  echo "ERROR: cursor-server binary not found under /root/.cursor-server/bin/linux-x64" >&2
  exit 1
fi
echo "Using cursor-server: $CURSOR_SERVER"

# Remove any existing ms-toolsai.jupyter installs (incompatible versions etc.)
echo "Removing existing Jupyter extension installs..."
rm -rf "$EXTENSIONS_DIR"/ms-toolsai.jupyter-*/

# Download the VSIX (--compressed handles gzip-encoded responses from the marketplace)
echo "Downloading ms-toolsai.jupyter ${EXTENSION_VERSION}..."
curl -L --compressed -o "$VSIX_TMP" "$VSIX_URL"

# Quick sanity check: VSIX files are zip files starting with PK (0x504b)
MAGIC=$(python3 -c "
with open('$VSIX_TMP', 'rb') as f:
    print(f.read(2).hex())
")
if [[ "$MAGIC" != "504b" ]]; then
  echo "ERROR: Downloaded file does not look like a valid VSIX (zip). Got magic bytes: $MAGIC" >&2
  rm -f "$VSIX_TMP"
  exit 1
fi



# Install
echo "Installing extension..."
# "$CURSOR_SERVER" --install-extension ms-python.python
# "$CURSOR_SERVER" --install-extension anysphere.cursorpyright
# "$CURSOR_SERVER" --install-extension Anthropic.claude-code
"$CURSOR_SERVER" --install-extension "$VSIX_TMP"

rm -f "$VSIX_TMP"
echo "Done! Reload the Cursor window (Ctrl+Shift+P → 'Developer: Reload Window') to activate."
