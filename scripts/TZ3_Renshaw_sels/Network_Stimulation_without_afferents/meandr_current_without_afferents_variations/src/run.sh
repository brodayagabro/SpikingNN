#!/bin/bash
set -euo pipefail

# Pipeline configuration
CONFIG_FILE="config.json"
GEN_SCRIPT="gen_cfg_meandr_pulse_without_afferents_rainshow_sells.py"
SIM_SCRIPT="calc_meandr_pulse_without_afferents_rainshow_sells.py"

# Parse command-line arguments
CLEANUP=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup|-c)
            CLEANUP=true
            shift
            ;;
        *)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Unknown option '$1'" >&2
            echo "Usage: bash $0 [--cleanup|-c]" >&2
            exit 1
            ;;
    esac
done

# Ensure script is running in bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: This script requires bash. Run with: bash $0" >&2
    exit 1
fi

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Pipeline execution started."

# 1. Verify configuration file
if [[ ! -f "$CONFIG_FILE" ]]; then
    log "ERROR: Configuration file '$CONFIG_FILE' not found in the current directory."
    exit 1
fi

# Extract parameters from JSON safely
CSV_FILE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('output_filename', 'cfg.csv'))")
ARCHIVE_NAME=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('archive_name', 'experiment_data_$(date +%Y%m%d_%H%M%S)'))")

# 2. Generate configuration
log "Step 1: Generating configuration file '$CSV_FILE'..."
python3 "$GEN_SCRIPT" "$CONFIG_FILE"

if [[ ! -s "$CSV_FILE" ]]; then
    log "ERROR: Configuration file '$CSV_FILE' was not created or is empty."
    exit 1
fi
log "Configuration generated successfully."

# 3. Run computations
log "Step 2: Starting computations..."
python3 "$SIM_SCRIPT" --config "$CSV_FILE"
log "Computations completed successfully."

# 4. Archive results
log "Step 3: Archiving '../data' directory..."
if [[ -d "../data" ]]; then
    # Verify 7-Zip is available
    if ! command -v 7z &>/dev/null; then
        log "ERROR: '7z' command not found. Install it with: sudo apt install p7zip-full"
        exit 1
    fi

    # Use subshell to temporarily change directory. This ensures:
    # 1. The archive is created in the parent directory (../)
    # 2. The archive contains only 'data/' without leading '../' paths
    (cd .. && 7z a -mx9 "${ARCHIVE_NAME}.zip" data)
    
    if [[ -f "../${ARCHIVE_NAME}.zip" ]]; then
        ARCHIVE_SIZE=$(du -h "../${ARCHIVE_NAME}.zip" | awk '{print $1}')
        log "Archive created at '../${ARCHIVE_NAME}.zip' (Size: ${ARCHIVE_SIZE})"
        
        # Conditional cleanup
        if [[ "$CLEANUP" == true ]]; then
            log "Cleanup flag detected. Removing '../data' directory..."
            rm -rf ../data
            if [[ ! -d "../data" ]]; then
                log "Cleanup completed: '../data' successfully removed."
            else
                log "ERROR: Failed to remove '../data' directory."
                exit 1
            fi
        fi
    else
        log "ERROR: 7-Zip archiving failed. Archive file was not created."
        exit 1
    fi
else
    log "WARNING: Directory '../data' not found. Skipping archiving step."
fi

log "Pipeline execution finished successfully."
