#!/bin/bash
# RAPPEL:
#
# Arguments positionnels:
#   PREDICTIONS_DIR:  Chemin vers le dossier contenant les sous-dossiers de predictions + ref (ex: forecasting/run_009/predictions)
#
# Options transmises à ipol_kervrann.py:
#   --scale INT       Nombre d'échelles          (défaut: 3)
#   --b     INT       Côté de la fenêtre locale  (défaut: 3)
#   --B     INT       Côté de la fenêtre search  (défaut: 3)
#   --metric STR      Mesure de dissimilarité    (défaut: lin)
#                     Choix: corr | rho | mult | zncc | lin
#   --epsilon FLOAT   Seuil de fausses alarmes   (défaut: 1.0)
#   --sigma FLOAT     Sigma du filtre gaussien   (défaut: 0.8)
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <PREDICTIONS_DIR> [--scale N] [--b N] [--metric STR] ..."
    exit 1
fi

PREDICTIONS_DIR="$1"
shift

if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "[ERREUR] Dossier introuvable : $PREDICTIONS_DIR"
    exit 1
fi

SCRIPT="change_detection/kervrann/ipol_kervrann.py"

TOTAL=0
DONE=0
ERRORS=0

TOTAL=$(find "$PREDICTIONS_DIR" -name "*_pred.tif" | wc -l)
echo " Paires trouvées : $TOTAL"
echo ""

find "$PREDICTIONS_DIR" -name "*_pred.tif" | sort | while read -r PRED_FILE; do
    # déduit le fichier ref correspondant
    REF_FILE="${PRED_FILE/_pred.tif/_ref.tif}"

    if [ ! -f "$REF_FILE" ]; then
        echo "[SKIP]  Pas de ref pour : $(basename "$PRED_FILE")"
        continue
    fi

    # construit le dossier de sortie en miroir de la structure d'entrée
    REL_PATH="${PRED_FILE#$PREDICTIONS_DIR/}" # chemin relatif
    STEM="${REL_PATH%_pred.tif}"  # sans suffixe _pred.tif
    OUTDIR="$(dirname "$PREDICTIONS_DIR")/change_detection/${STEM}"

    mkdir -p "$OUTDIR"

    DONE=$((DONE + 1))
    echo "[${DONE}/${TOTAL}] $(basename "$PRED_FILE")"
    echo "         pred : $PRED_FILE"
    echo "         ref  : $REF_FILE"
    echo "         out  : $OUTDIR"

    # détection de changements
    if python "$SCRIPT" \
        --image1 "$PRED_FILE" \
        --image2 "$REF_FILE" \
        --dirout "$OUTDIR" \
        "$@"; then
        echo "         [OK]"
    else
        echo "         [ERREUR] — voir logs ci-dessus"
        ERRORS=$((ERRORS + 1))
    fi

    echo ""
done

echo "Résultats dans: $(dirname "$PREDICTIONS_DIR")/change_detection/"