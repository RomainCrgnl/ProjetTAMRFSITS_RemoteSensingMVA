import os
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def run_naive_detection(run_dir):

    SCRIPT = os.path.join("change_detection","kervrann", "ipol_kervrann.py")
   
    run_path = Path(run_dir)
    pred_dir = run_path / "predictions"
    
    if not pred_dir.exists():
        print(f"Erreur: Le dossier 'predictions' n'existe pas dans {run_dir}")
        return

    # 1. Trouver toutes les prédictions en vrac avec rglob (récursif) pour .tif et .tiff
    ref_files = list(pred_dir.rglob("*_ref.tif*"))
    
    if not ref_files:
        print(f"Aucun fichier *_ref.tif(f) trouvé dans {pred_dir} ou ses sous-dossiers.")
        return
    
    for i, ref_file in enumerate(ref_files) :

        if i != 0:
            
            ext = ".tiff" if ref_file.name.endswith(".tiff") else ".tif"
            base_name = ref_file.name.replace(f"_ref{ext}", "")

            OUTDIR = os.path.join(run_dir, "naive_change_detection", base_name)
            os.makedirs(OUTDIR, exist_ok=True)

            os.system(f"python {SCRIPT} --image1 {ref_file} --image2 {ref_files[i-1]} --dirout {OUTDIR}")
        

if __name__ == "__main__":
    name = "run_009"
    run_dir = os.path.join(".", "all", name)
    run_naive_detection(run_dir)