#!/usr/bin/env bash
# Makes the live Q&A's port (8501) public, so the class can open it without signing in to GitHub,
# or private again afterwards. Run it in the codespace's terminal, with the app running:
#   bash .devcontainer/share-live-qa.sh          # public, for the session
#   bash .devcontainer/share-live-qa.sh private  # private again, straight after
# A public port means anyone with the link can use the app, so make it public only while you present.
set -euo pipefail

VISIBILITY="${1:-public}"
case "$VISIBILITY" in
  public|private|org) ;;
  *) echo "Usage: bash .devcontainer/share-live-qa.sh [public|private|org]"; exit 1 ;;
esac

if [ -z "${CODESPACE_NAME:-}" ]; then
  echo "This only works in a GitHub codespace (CODESPACE_NAME isn't set)."
  exit 1
fi
if ! command -v gh > /dev/null; then
  echo "The GitHub CLI (gh) isn't installed. Use the Ports tab instead: right-click port 8501 >"
  echo "Port Visibility > Public."
  exit 1
fi

gh codespace ports visibility "8501:${VISIBILITY}" -c "$CODESPACE_NAME"
echo "Port 8501 is now ${VISIBILITY}. Share this address:"
echo "  https://${CODESPACE_NAME}-8501.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
