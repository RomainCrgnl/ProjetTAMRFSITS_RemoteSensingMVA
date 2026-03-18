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
    if img_data is None:
        return None
    rgb = np.stack([img_data[2], img_data[1], img_data[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    return np.clip(rgb / 0.3, 0, 1)

def get_cd_display(img_data):
    """Prépare l'image de Change Detection pour l'affichage."""
    if img_data is None:
        return None
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
    naive_cd_dir = run_path / "naive_change_detection"
    
    if not pred_dir.exists() or not cd_dir.exists():
        print(f"Erreur: Assurez-vous que les dossiers 'predictions' et 'change_detection' existent dans {run_dir}")
        return

    # 1. Regrouper tous les fichiers disponibles
    scenes = {}
    
    for p in pred_dir.rglob("*.tif*"):
        if "_pred" in p.name or "_ref" in p.name:
            ext = ".tiff" if p.name.endswith(".tiff") else ".tif"
            is_pred = "_pred" in p.name
            base_name = p.name.replace(f"_pred{ext}", "").replace(f"_ref{ext}", "")
            
            parts = base_name.split('_')
            date_str = parts[1] if len(parts) > 1 else "Unknown"
            
            # Reconstruire un identifiant de localisation (ignorer la date)
            if len(parts) > 1:
                loc_id = parts[0] + "_" + "_".join(parts[2:])
            else:
                loc_id = base_name
            
            if loc_id not in scenes:
                scenes[loc_id] = {}
            if date_str not in scenes[loc_id]:
                scenes[loc_id][date_str] = {'ref': None, 'pred': None, 'cd': None, 'naive_cd': None, 'base_name': base_name}
            
            if is_pred:
                scenes[loc_id][date_str]['pred'] = p
            else:
                scenes[loc_id][date_str]['ref'] = p

    # Associer les fichiers Change Detection et Naive Change Detection
    for loc_id, dates in scenes.items():
        for date_str, data in dates.items():
            base_name = data['base_name']
            
            # CD Normal
            cd_file = cd_dir / base_name / "huv_final_cube.tif"
            if not cd_file.exists():
                possible_cds = list(cd_dir.rglob(f"*{base_name}*/huv_final_cube.tif"))
                if possible_cds:
                    cd_file = possible_cds[0]
            if cd_file.exists():
                data['cd'] = cd_file
                
            # Naive CD (Même structure que Change Detection)
            if naive_cd_dir.exists():
                naive_cd_file = naive_cd_dir / base_name / "huv_final_cube.tif"
                if not naive_cd_file.exists():
                    possible_naive_cds = list(naive_cd_dir.rglob(f"*{base_name}*/huv_final_cube.tif"))
                    if possible_naive_cds:
                        naive_cd_file = possible_naive_cds[0]
                if naive_cd_file.exists():
                    data['naive_cd'] = naive_cd_file

    # 2. Construire la timeline
    max_dates_per_loc = max([len(dates) for dates in scenes.values()]) if scenes else 0
    timeline = []
    
    if max_dates_per_loc == 1:
        # Fallback : S'il n'y a qu'une date par scène identifiée, on crée une liste chronologique globale
        print("Fallback activé : Les localisations n'ont qu'une seule date. Navigation globale chronologique.")
        flat_scenes = []
        for loc_id, dates in scenes.items():
            for date_str, data in dates.items():
                data['loc_id'] = loc_id
                data['date'] = date_str
                flat_scenes.append(data)
                
        flat_scenes.sort(key=lambda x: x['date'])
        
        for i, data in enumerate(flat_scenes):
            prev_data = flat_scenes[i-1] if i > 0 else None
            timeline.append({
                'patch_id': data['loc_id'],
                'date': data['date'],
                'current': data,
                'prev': prev_data
            })
    else:
        # Normal : Regroupement multi-dates respecté
        for loc_id in sorted(scenes.keys()):
            sorted_dates = sorted(scenes[loc_id].keys())
            for i, date_str in enumerate(sorted_dates):
                current_data = scenes[loc_id][date_str]
                prev_data = scenes[loc_id][sorted_dates[i-1]] if i > 0 else None
                
                timeline.append({
                    'patch_id': loc_id,
                    'date': date_str,
                    'current': current_data,
                    'prev': prev_data
                })
    
    if not timeline:
        print("Aucun fichier pertinent trouvé.")
        return
        
    print(f"Trouvé {len(timeline)} scènes dans la timeline temporelle. Lancement du viewer...")

    # 3. Configuration de l'interface (2 lignes, 3 colonnes)
    if 'left' in plt.rcParams['keymap.back']:
        plt.rcParams['keymap.back'].remove('left')
    if 'right' in plt.rcParams['keymap.forward']:
        plt.rcParams['keymap.forward'].remove('right')

    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Ligne 1
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.set_title("Référence (t)", fontsize=14)
    ax_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_pred = fig.add_subplot(gs[0, 1], sharex=ax_ref, sharey=ax_ref) 
    ax_pred.set_title("Prédiction (t)", fontsize=14)
    ax_pred.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_cd = fig.add_subplot(gs[0, 2], sharex=ax_ref, sharey=ax_ref) 
    ax_cd.set_title("Change Detection (HUV Cube)", fontsize=14)
    ax_cd.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Ligne 2
    ax_ref2 = fig.add_subplot(gs[1, 0], sharex=ax_ref, sharey=ax_ref)
    ax_ref2.set_title("Référence (t) (Rappel)", fontsize=14)
    ax_ref2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_prev_ref = fig.add_subplot(gs[1, 1], sharex=ax_ref, sharey=ax_ref)
    ax_prev_ref.set_title("Référence (t-1)", fontsize=14)
    ax_prev_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_naive_cd = fig.add_subplot(gs[1, 2], sharex=ax_ref, sharey=ax_ref)
    ax_naive_cd.set_title("Naive Change Detection", fontsize=14)
    ax_naive_cd.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    im_refs = {'ref': None, 'pred': None, 'cd': None, 'ref2': None, 'prev_ref': None, 'naive_cd': None}

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # 4. Gestion de l'état et du cache
    state = {'current_index': 0}
    cache = {} 
    executor = ThreadPoolExecutor(max_workers=2)

    def load_data(item):
        cur_ref = process_tiff(item['current']['ref']) if item['current']['ref'] else None
        cur_pred = process_tiff(item['current']['pred']) if item['current']['pred'] else None
        cur_cd = process_cd_tiff(item['current']['cd']) if item['current']['cd'] else None
        cur_naive_cd = process_cd_tiff(item['current']['naive_cd']) if item['current']['naive_cd'] else None
        
        prev_ref = None
        if item['prev'] and item['prev']['ref']:
            prev_ref = process_tiff(item['prev']['ref'])
            
        return cur_ref, cur_pred, cur_cd, prev_ref, cur_naive_cd

    def manage_cache(current_idx):
        target_indices = [current_idx]
        if current_idx > 0: target_indices.append(current_idx - 1)
        if current_idx < len(timeline) - 1: target_indices.append(current_idx + 1)
            
        for idx in target_indices:
            if idx not in cache:
                cache[idx] = executor.submit(load_data, timeline[idx])
                
        keys_to_remove = [k for k in cache.keys() if abs(k - current_idx) > 2]
        for k in keys_to_remove:
            cache[k].cancel()
            del cache[k]

    def update_display():
        idx = state['current_index']
        item = timeline[idx]
        
        raw_date_str = item['date']
        try:
            parsed_date = datetime.strptime(raw_date_str, "%Y-%m-%d").date().strftime('%Y-%m-%d')
        except ValueError:
            parsed_date = raw_date_str
            
        prev_date_str = "N/A"
        if item['prev']:
            try:
                prev_date_str = datetime.strptime(item['prev']['date'], "%Y-%m-%d").date().strftime('%Y-%m-%d')
            except ValueError:
                prev_date_str = item['prev']['date']

        fig.suptitle(f"Location ID: {item['patch_id']} | Date(t): {parsed_date} | Date(t-1): {prev_date_str} "
                     f"({idx + 1}/{len(timeline)})", 
                     fontsize=16, fontweight='bold')
        
        manage_cache(idx)
        
        cur_ref, cur_pred, cur_cd, prev_ref, cur_naive_cd = cache[idx].result()
        
        valid_img = cur_ref if cur_ref is not None else (cur_pred if cur_pred is not None else None)
        h, w = (256, 256)
        if valid_img is not None:
            if valid_img.ndim == 3:
                h, w = valid_img.shape[1], valid_img.shape[2]
            else:
                h, w = valid_img.shape[0], valid_img.shape[1]
                
        black_img_rgb = np.zeros((h, w, 3), dtype=np.float32)

        rgb_ref = get_rgb_image(cur_ref) if cur_ref is not None else black_img_rgb
        rgb_pred = get_rgb_image(cur_pred) if cur_pred is not None else black_img_rgb
        disp_cd = get_cd_display(cur_cd) if cur_cd is not None else black_img_rgb
        rgb_prev_ref = get_rgb_image(prev_ref) if prev_ref is not None else black_img_rgb
        
        # Traitement direct du dossier naive_change_detection
        disp_naive_cd = get_cd_display(cur_naive_cd) if cur_naive_cd is not None else black_img_rgb

        # Suppression systématique de l'affichage précédent pour éviter les conflits de dimensions
        if im_refs['ref'] is not None:
            try:
                im_refs['ref'].remove()
                im_refs['pred'].remove()
                im_refs['cd'].remove()
                im_refs['ref2'].remove()
                im_refs['prev_ref'].remove()
                im_refs['naive_cd'].remove()
            except Exception:
                pass

        im_refs['ref'] = ax_ref.imshow(rgb_ref)
        im_refs['pred'] = ax_pred.imshow(rgb_pred)
        im_refs['cd'] = ax_cd.imshow(disp_cd, cmap='viridis' if (cur_cd is not None and disp_cd.ndim == 2) else None)
        im_refs['ref2'] = ax_ref2.imshow(rgb_ref)
        im_refs['prev_ref'] = ax_prev_ref.imshow(rgb_prev_ref)
        im_refs['naive_cd'] = ax_naive_cd.imshow(disp_naive_cd, cmap='viridis' if (cur_naive_cd is not None and disp_naive_cd.ndim == 2) else None)
            
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == 'right':
            state['current_index'] = (state['current_index'] + 1) % len(timeline)
            update_display()
        elif event.key == 'left':
            state['current_index'] = (state['current_index'] - 1) % len(timeline)
            update_display()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_display()
    
    plt.show()

if __name__ == "__main__":
    name = "run_011"
    run_dir = os.path.join("..", "all", name)
    launch_comparison_viewer(run_dir)