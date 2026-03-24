import os
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def process_tiff(file_path):
    """Lit le TIFF (Pred/Ref) dans sa résolution native."""
    with rasterio.open(file_path) as src:
        img_data = src.read()

    img_data = img_data.astype(np.float32) 
    img_data[img_data == -10000] = np.nan
    img_data /= 10_000
    return img_data

def process_cd_tiff(file_path):
    """Lit le TIFF de Change Detection (huv_final_cube.tif)."""
    with rasterio.open(file_path) as src:
        img_data = src.read()
    return img_data.astype(np.float32)

def get_rgb_image(img_data):
    """Extrait le Rouge(B4), Vert(B3), Bleu(B2) et applique un stretch de contraste."""
    rgb = np.stack([img_data[2], img_data[1], img_data[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    return np.clip(rgb / 0.3, 0, 1)

def get_cd_display(img_data):
    """Prépare l'image de Change Detection pour l'affichage."""
    img_data = np.nan_to_num(img_data, nan=0.0)
    
    if img_data.shape[0] >= 3:
        display_img = np.stack([img_data[0], img_data[1], img_data[2]], axis=-1)
    else:
        display_img = img_data[0]
        
    img_min = display_img.min()
    img_max = display_img.max()
    ptp = img_max - img_min
    
    if ptp > 0:
        display_img = (display_img - img_min) / ptp
        
    return display_img

def extract_date_from_filename(filename):
    """Extrait la date du format '32_2022-02-02_hr_mae_..._pred.tif(f)'"""
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]
    return "Unknown"

def launch_comparison_viewer(run_dir):
    run_path = Path(run_dir)
    pred_dir = run_path / "predictions"
    cd_dir = run_path / "change_detection"
    
    if not pred_dir.exists() or not cd_dir.exists():
        print(f"Erreur: Assurez-vous que les dossiers 'predictions' et 'change_detection' existent dans {run_dir}")
        return

    # 1. Trouver toutes les prédictions en vrac avec rglob (récursif) pour .tif et .tiff
    pred_files = list(pred_dir.rglob("*_pred.tif*"))
    
    if not pred_files:
        print(f"Aucun fichier *_pred.tif(f) trouvé dans {pred_dir} ou ses sous-dossiers.")
        return

    file_triplets = []
    for pred_file in pred_files:
        # Gérer l'extension exacte (.tif ou .tiff)
        ext = ".tiff" if pred_file.name.endswith(".tiff") else ".tif"
        base_name = pred_file.name.replace(f"_pred{ext}", "")
        
        # La ref est dans le même dossier que la pred
        ref_file = pred_file.parent / f"{base_name}_ref{ext}"
        
        # Le Change Detection. On tente d'abord le chemin direct.
        cd_file = cd_dir / base_name / "huv_final_cube.tif"
        
        # Si non trouvé, on cherche récursivement dans cd_dir au cas où l'arborescence diffère légèrement
        if not cd_file.exists():
            possible_cds = list(cd_dir.rglob(f"*{base_name}*/huv_final_cube.tif"))
            if possible_cds:
                cd_file = possible_cds[0]
        
        if ref_file.exists() and cd_file.exists():
            file_triplets.append((pred_file, ref_file, cd_file))
        else:
            if not ref_file.exists():
                print(f"Attention: Référence manquante pour {base_name} dans {pred_file.parent}")
            if not cd_file.exists():
                print(f"Attention: Change Detection (huv_final_cube.tif) manquant pour le dossier {base_name}")

    # Trier les triplets par date
    file_triplets.sort(key=lambda x: extract_date_from_filename(x[0].name))
    
    if not file_triplets:
        print("Aucun triplet complet (Pred, Ref, CD) n'a été trouvé.")
        return
        
    print(f"Trouvé {len(file_triplets)} triplets d'images. Lancement du viewer...")

    # 2. Configuration de l'interface (3 images côte à côte)
    if 'left' in plt.rcParams['keymap.back']:
        plt.rcParams['keymap.back'].remove('left')
    if 'right' in plt.rcParams['keymap.forward']:
        plt.rcParams['keymap.forward'].remove('right')

    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.set_title("Référence", fontsize=14)
    ax_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_pred = fig.add_subplot(gs[0, 1], sharex=ax_ref, sharey=ax_ref) 
    ax_pred.set_title("Prédiction", fontsize=14)
    ax_pred.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_cd = fig.add_subplot(gs[0, 2], sharex=ax_ref, sharey=ax_ref) 
    ax_cd.set_title("Change Detection (HUV Cube)", fontsize=14)
    ax_cd.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    im_ref = None
    im_pred = None
    im_cd = None

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # 3. Gestion de l'état et du cache
    state = {'current_index': 0}
    cache = {} 
    executor = ThreadPoolExecutor(max_workers=2)

    def load_triplet(pred_path, ref_path, cd_path):
        return process_tiff(pred_path), process_tiff(ref_path), process_cd_tiff(cd_path)

    def manage_cache(current_idx):
        target_indices = [current_idx]
        if current_idx > 0: target_indices.append(current_idx - 1)
        if current_idx < len(file_triplets) - 1: target_indices.append(current_idx + 1)
            
        for idx in target_indices:
            if idx not in cache:
                pred_path, ref_path, cd_path = file_triplets[idx]
                cache[idx] = executor.submit(load_triplet, pred_path, ref_path, cd_path)
                
        keys_to_remove = [k for k in cache.keys() if abs(k - current_idx) > 2]
        for k in keys_to_remove:
            cache[k].cancel()
            del cache[k]

    def update_display():
        nonlocal im_ref, im_pred, im_cd
        
        idx = state['current_index']
        pred_file, _, _ = file_triplets[idx]
        
        raw_date_str = extract_date_from_filename(pred_file.name)
        try:
            parsed_date = datetime.strptime(raw_date_str, "%Y-%m-%d").date().strftime('%Y-%m-%d')
        except ValueError:
            parsed_date = raw_date_str

        fig.suptitle(f"{parsed_date}", fontsize=20, fontweight='bold')
        
        manage_cache(idx)
        
        pred_data, ref_data, cd_data = cache[idx].result()
        
        rgb_pred = get_rgb_image(pred_data)
        rgb_ref = get_rgb_image(ref_data)
        disp_cd = get_cd_display(cd_data)
        
        h, w = rgb_pred.shape[0], rgb_pred.shape[1]

        if im_ref is None or im_pred is None or im_cd is None:
            im_ref = ax_ref.imshow(rgb_ref)
            im_pred = ax_pred.imshow(rgb_pred)
            im_cd = ax_cd.imshow(disp_cd, cmap='viridis' if disp_cd.ndim == 2 else None)
        else:
            im_ref.set_data(rgb_ref)
            im_pred.set_data(rgb_pred)
            im_cd.set_data(disp_cd)
            
            im_ref.set_extent([0, w, h, 0])
            im_pred.set_extent([0, w, h, 0])
            im_cd.set_extent([0, w, h, 0])
            
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == 'right':
            state['current_index'] = (state['current_index'] + 1) % len(file_triplets)
            update_display()
        elif event.key == 'left':
            state['current_index'] = (state['current_index'] - 1) % len(file_triplets)
            update_display()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_display()
    
    plt.show()


if __name__ == "__main__":
    name = "run_009"
    run_dir = os.path.join("..", "all", name)
    launch_comparison_viewer(run_dir)