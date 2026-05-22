# Shared library for serving_agent entry-point scripts.
# Source this as the first thing in any script:
#   source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"

# Resolve to the scripts/ directory from this file's location in scripts/lib/.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENV_FILE="$_COMMON_DIR/../env"

if [ ! -f "$_ENV_FILE" ]; then
    echo "ERROR: Configuration file not found: $_ENV_FILE" >&2
    echo "" >&2
    echo "  cp $_COMMON_DIR/../env.example $_ENV_FILE" >&2
    echo "  # Then edit $_ENV_FILE with your local paths" >&2
    echo "" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$_ENV_FILE"
