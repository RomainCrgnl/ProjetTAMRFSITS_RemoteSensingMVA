#!/bin/bash
# =============================================================================
# run_change_detection.sh
# Lance la détection de changements (Kervrann et al.) sur toutes les paires
# pred/ref d'un dossier de prédictions.
#
# Usage:
#   bash run_change_detection.sh <PREDICTIONS_DIR> [OPTIONS]
#
# Exemple:
#   bash run_change_detection.sh forecasting/run_009/predictions \
#       --scale 3 --b 3 --metric lin --epsilon 1.0
#
# Arguments positionnels:
#   PREDICTIONS_DIR   Chemin vers le dossier contenant les sous-dossiers
#                     de tuiles (ex: forecasting/run_009/predictions)
#
# Options transmises à ipol_kervrann.py (toutes optionnelles):
#   --scale INT       Nombre d'échelles         (défaut: 3)
#   --b     INT       Côté de la fenêtre locale  (défaut: 3)
#   --B     INT       Côté de la fenêtre search  (défaut: 3)
#   --metric STR      Mesure de dissimilarité    (défaut: lin)
#                     Choix: corr | rho | mult | zncc | lin
#   --epsilon FLOAT   Seuil de fausses alarmes   (défaut: 1.0)
#   --sigma FLOAT     Sigma du filtre gaussien   (défaut: 0.8)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Arguments
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <PREDICTIONS_DIR> [--scale N] [--b N] [--metric STR] ..."
    exit 1
fi

PREDICTIONS_DIR="$1"
shift  # le reste ($@) sera transmis directement à ipol_kervrann.py

if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "[ERREUR] Dossier introuvable : $PREDICTIONS_DIR"
    exit 1
fi

# Chemin vers le script Python (modifiable si besoin)
SCRIPT="change_detection/kervrann/ipol_kervrann.py"

if [ ! -f "$SCRIPT" ]; then
    echo "[ERREUR] Script Python introuvable : $SCRIPT"
    echo "         Définir la variable SCRIPT_PATH si nécessaire."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Découverte des paires pred / ref
# ---------------------------------------------------------------------------
# Structure attendue :
#   PREDICTIONS_DIR/
#     <tile>/
#       <model_dir>/
#         <DOY>_<DATE>_..._pred.tif   ← image prédite
#         <DOY>_<DATE>_..._ref.tif    ← image de référence
# ---------------------------------------------------------------------------

echo "============================================================"
echo " Détection de changements — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Dossier d'entrée : $PREDICTIONS_DIR"
echo "============================================================"

TOTAL=0
DONE=0
ERRORS=0

# Compte d'abord pour l'affichage de progression
TOTAL=$(find "$PREDICTIONS_DIR" -name "*_pred.tif" | wc -l)
echo " Paires trouvées : $TOTAL"
echo ""

# ---------------------------------------------------------------------------
# 3. Boucle principale
# ---------------------------------------------------------------------------
find "$PREDICTIONS_DIR" -name "*_pred.tif" | sort | while read -r PRED_FILE; do

    # Déduit le fichier ref correspondant
    REF_FILE="${PRED_FILE/_pred.tif/_ref.tif}"

    if [ ! -f "$REF_FILE" ]; then
        echo "[SKIP]  Pas de ref pour : $(basename "$PRED_FILE")"
        continue
    fi

    # Construit le dossier de sortie en miroir de la structure d'entrée
    # Ex: predictions/30SWH_24/hr_mae_.../107_2022-04-18_..._pred.tif
    #  -> change_detection/30SWH_24/hr_mae_.../107_2022-04-18_...
    REL_PATH="${PRED_FILE#$PREDICTIONS_DIR/}"           # chemin relatif
    STEM="${REL_PATH%_pred.tif}"                        # sans suffixe _pred.tif
    OUTDIR="$(dirname "$PREDICTIONS_DIR")/change_detection/${STEM}"

    mkdir -p "$OUTDIR"

    DONE=$((DONE + 1))
    echo "[${DONE}/${TOTAL}] $(basename "$PRED_FILE")"
    echo "         pred : $PRED_FILE"
    echo "         ref  : $REF_FILE"
    echo "         out  : $OUTDIR"

    # Lancement de la détection de changements
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

# ---------------------------------------------------------------------------
# 4. Résumé
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Terminé : $DONE paires traitées, $ERRORS erreur(s)."
echo " Résultats dans : $(dirname "$PREDICTIONS_DIR")/change_detection/"
echo "============================================================"