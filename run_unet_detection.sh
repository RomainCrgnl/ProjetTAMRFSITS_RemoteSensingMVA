
#!/bin/bash

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <experiment_path> [output_base]"
    echo ""
    echo "Example:"
    echo "  $0 '30SWH_24_c5_g1/predictions/30SWH_24/hr_mae_CUSTOM_FORECAST_50_318.0' '30SWH_24_c5_g1/output_unet'"
    echo ""
    echo "If output_base is not provided, it will default to <experiment_parent>/output_unet"
    exit 1
fi

EXPERIMENT_PATH="$1"
python ${SCRIPT_DIR}/change_detection/Unet/run_change_detection.py "$EXPERIMENT_PATH"