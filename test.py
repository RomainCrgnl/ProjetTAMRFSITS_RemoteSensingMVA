import os
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

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

def get_last_pred_ref_pair(base_dir):
    """Finds the chronological last prediction and its reference in the directory."""
    base_path = Path(base_dir)
    pred_files = list(base_path.glob("*_pred.tif"))

    if not pred_files:
        return None, None, None

    # Sort files chronologically based on the date in the filename
    # Example filename: '0_2022-12-21_hr_clr_FORECAST_50_355.0_pred.tif'
    pred_files.sort(key=lambda x: x.name.split('_')[1])

    last_pred = pred_files[-1]
    last_ref = last_pred.parent / last_pred.name.replace("_pred.tif", "_ref.tif")
    date_str = last_pred.name.split('_')[1]

    return last_pred, last_ref, date_str

def launch_comparator_viewer(base_dir):
    pred_file, ref_file, date_str = get_last_pred_ref_pair(base_dir)

    if not pred_file:
        print(f"No prediction files found in {base_dir}")
        return
    if not ref_file.exists():
        print(f"Reference file missing for {pred_file.name}")
        return

    parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    print(f"Loading last date: {parsed_date}. Mode: Blink Comparator.")

    band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B5 (Red Edge 1)',
                  'B6 (Red Edge 2)', 'B7 (Red Edge 3)', 'B8 (NIR)',
                  'B8A (Narrow NIR)', 'B11 (SWIR 1)', 'B12 (SWIR 2)']

    # Disable default matplotlib keymaps for arrows
    if 'left' in plt.rcParams['keymap.back']:
        plt.rcParams['keymap.back'].remove('left')
    if 'right' in plt.rcParams['keymap.forward']:
        plt.rcParams['keymap.forward'].remove('right')
    if 'space' in plt.rcParams['keymap.fullscreen']:
        plt.rcParams['keymap.fullscreen'].remove('space')

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    im_objects = []
    for i in range(10):
        # CRITICAL TRICK: Shared coordinate space for seamless LoD
        im = axes[i].imshow(np.zeros((330, 330)), cmap='gray', vmin=0, vmax=0.3, extent=[0, 990, 990, 0])
        axes[i].set_title(band_names[i])
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        im_objects.append(im)

        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Leave room for the big title

    # Pre-load low-res into memory for instant toggling
    print("Pre-caching low-res images...")
    cache = {
        'ref_low': process_tiff(ref_file, low_res=False),
        'pred_low': process_tiff(pred_file, low_res=False),
        'ref_high': None,
        'pred_high': None
    }

    # State management
    files = {'ref': ref_file, 'pred': pred_file}
    state = {'current_view': 'ref', 'is_high_res': False}

    def is_zoomed_in():
        xlim = axes[0].get_xlim()
        return abs(xlim[1] - xlim[0]) < 950

    def update_display():
        view = state['current_view']
        zoomed = is_zoomed_in()

        # Determine which resolution to load based on zoom
        if zoomed:
            if cache[f'{view}_high'] is None:
                print(f"Loading native resolution for {view.upper()}...")
                cache[f'{view}_high'] = process_tiff(files[view], low_res=False)
            img_data = cache[f'{view}_high']
            state['is_high_res'] = True
        else:
            img_data = cache[f'{view}_low']
            state['is_high_res'] = False

        for i in range(10):
            im_objects[i].set_data(img_data[i])

        fig.canvas.draw_idle()
        update_title()

    def update_title():
        view_name = "GROUND TRUTH REFERENCE" if state['current_view'] == 'ref' else "MODEL PREDICTION"
        color = "green" if state['current_view'] == 'ref' else "red"
        res_text = "NATIVE 10m" if state['is_high_res'] else "PROXY (Zoom to enhance)"

        fig.suptitle(
            f"Date: {parsed_date.strftime('%Y-%m-%d')} | Resolution: {res_text}\n"
            f"Currently showing: {view_name} (Press LEFT/RIGHT/SPACE to toggle)",
            fontsize=16, fontweight='bold', color=color
        )

    def on_xlim_changed(ax):
        zoomed = is_zoomed_in()
        if zoomed and not state['is_high_res']:
            update_display()
        elif not zoomed and state['is_high_res']:
            update_display()

    def on_key_press(event):
        # Toggle between Ref and Pred on keypress
        if event.key in ['right', 'left', 'space']:
            state['current_view'] = 'pred' if state['current_view'] == 'ref' else 'ref'
            update_display()

    for ax in axes:
        ax.callbacks.connect('xlim_changed', on_xlim_changed)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Initialize the first view
    update_display()
    plt.show()

if __name__ == "__main__":
    # IMPORTANT: Update this path to point to the folder containing your _pred.tif and _ref.tif files
    # For example, after running with --forecast_doy_start 355
    output_dir = os.path.join(".", "gap_filling", "tamrfsits", "predictions", "test", "hr_clr_FORECAST_50_355.0")
    launch_comparator_viewer(output_dir)