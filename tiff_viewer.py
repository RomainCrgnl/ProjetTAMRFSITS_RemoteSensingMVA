import os
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def process_tiff(file_path, low_res=True):
    """Reads TIFF. Returns 330x330 if low_res=True, else full 990x990."""
    with rasterio.open(file_path) as src:
        if low_res:
            out_shape = (src.count, src.height // 3, src.width // 3)
            img_data = src.read(out_shape=out_shape, resampling=Resampling.nearest)
        else:
            img_data = src.read()
            
    img_data = img_data.astype(np.float32) 
    img_data[img_data == -10000] = np.nan
    img_data /= 10000.0                   
    return img_data

def launch_timeseries_viewer(base_dir):
    base_path = Path(base_dir)
    tif_files = list(base_path.rglob("sentinel2_bands_*.tif"))
    
    if not tif_files:
        print(f"No TIFF files found in {base_dir}")
        return

    tif_files.sort(key=lambda x: x.name)
    print(f"Found {len(tif_files)} dates. Loading dynamic LoD viewer...")

    band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B5 (Red Edge 1)', 
                  'B6 (Red Edge 2)', 'B7 (Red Edge 3)', 'B8 (NIR)', 
                  'B8A (Narrow NIR)', 'B11 (SWIR 1)', 'B12 (SWIR 2)']
    
    if 'left' in plt.rcParams['keymap.back']:
        plt.rcParams['keymap.back'].remove('left')
    if 'right' in plt.rcParams['keymap.forward']:
        plt.rcParams['keymap.forward'].remove('right')

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    im_objects = []
    for i in range(10):
        # CRITICAL TRICK: We set extent=[0, 990, 990, 0]. 
        # This forces both low-res (330) and high-res (990) arrays to share the exact same coordinate space!
        im = axes[i].imshow(np.zeros((330, 330)), cmap='gray', vmin=0, vmax=0.3, extent=[0, 990, 990, 0])
        axes[i].set_title(band_names[i])
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        im_objects.append(im)
        
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # State management
    state = {'current_index': 0, 'is_high_res': False}
    cache = {} 
    executor = ThreadPoolExecutor(max_workers=2)

    def manage_cache(current_idx):
        """Keeps low-res images pre-loaded in the background."""
        target_indices = [current_idx]
        if current_idx > 0: target_indices.append(current_idx - 1)
        if current_idx < len(tif_files) - 1: target_indices.append(current_idx + 1)
            
        for idx in target_indices:
            if idx not in cache:
                cache[idx] = executor.submit(process_tiff, tif_files[idx], low_res=True)
                
        keys_to_remove = [k for k in cache.keys() if abs(k - current_idx) > 2]
        for k in keys_to_remove:
            cache[k].cancel()
            del cache[k]

    def is_zoomed_in():
        """Checks if the user is currently zoomed in."""
        xlim = axes[0].get_xlim()
        # Full width is 990. If the view width is smaller than 950, they zoomed.
        return abs(xlim[1] - xlim[0]) < 950

    def update_display():
        """Updates the image data when arrow keys are pressed."""
        idx = state['current_index']
        current_file = tif_files[idx]
        
        raw_date_str = current_file.stem.split('_')[-1] 
        parsed_date = datetime.strptime(raw_date_str, "%Y%m%d").date()
        fig.suptitle(f"Sentinel-2 | Date: {parsed_date.strftime('%Y-%m-%d')} "
                     f"({idx + 1}/{len(tif_files)}) | High-Res Mode: {state['is_high_res']}", 
                     fontsize=14, fontweight='bold')
        
        manage_cache(idx)
        
        # If navigating while already zoomed in, fetch high-res immediately
        if is_zoomed_in():
            img_data = process_tiff(current_file, low_res=False)
            state['is_high_res'] = True
        else:
            # If zoomed out, use the lightning-fast low-res cache
            img_data = cache[idx].result()
            state['is_high_res'] = False
            
        for i in range(10):
            im_objects[i].set_data(img_data[i])
            
        fig.canvas.draw_idle()

    def on_xlim_changed(ax):
        """Triggered automatically when the user zooms or pans."""
        zoomed = is_zoomed_in()
        
        # If they zoom IN and we are currently in low res: upgrade to high res!
        if zoomed and not state['is_high_res']:
            state['is_high_res'] = True
            print("Zoom detected! Enhancing to native 10m resolution...")
            
            current_file = tif_files[state['current_index']]
            img_data = process_tiff(current_file, low_res=False)
            
            for i in range(10): im_objects[i].set_data(img_data[i])
            fig.canvas.draw_idle()
            
            # Update title to show high-res is active
            update_title_only()
            
        # If they zoom OUT to full view and we are in high res: revert to low res for speed!
        elif not zoomed and state['is_high_res']:
            state['is_high_res'] = False
            print("Zoom reset! Reverting to proxy resolution for fast scrolling...")
            
            img_data = cache[state['current_index']].result()
            for i in range(10): im_objects[i].set_data(img_data[i])
            fig.canvas.draw_idle()
            update_title_only()

    def update_title_only():
        idx = state['current_index']
        raw_date_str = tif_files[idx].stem.split('_')[-1] 
        parsed_date = datetime.strptime(raw_date_str, "%Y%m%d").date()
        res_text = "Active" if state['is_high_res'] else "Inactive"
        fig.suptitle(f"Sentinel-2 | Date: {parsed_date.strftime('%Y-%m-%d')} "
                     f"({idx + 1}/{len(tif_files)}) | High-Res Mode: {res_text}", 
                     fontsize=14, fontweight='bold')

    def on_key_press(event):
        if event.key == 'right':
            state['current_index'] = (state['current_index'] + 1) % len(tif_files)
            update_display()
        elif event.key == 'left':
            state['current_index'] = (state['current_index'] - 1) % len(tif_files)
            update_display()

    # Bind the zoom detector to all axes
    for ax in axes:
        ax.callbacks.connect('xlim_changed', on_xlim_changed)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_display()
    
    plt.show()

if __name__ == "__main__":
    dataset_dir = os.path.join(".", "dataset", "test", "31TCJ_12", "sentinel2")
    launch_timeseries_viewer(dataset_dir)