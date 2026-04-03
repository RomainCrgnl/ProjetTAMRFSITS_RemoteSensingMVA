import os
import glob
import shutil
import argparse
import rasterio
import numpy as np
from PIL import Image

def convert_tif_to_png(file_path, out_path, is_raw_mask=False):
    """
    Lit le TIF entier et le convertit en PNG.
    Si is_raw_mask=True, on ne touche pas aux valeurs (pas de normalisation Sentinel-2).
    """
    with rasterio.open(file_path) as src:
        img_data = src.read()

    img_data = img_data.astype(np.float32)

    if not is_raw_mask:
        # TRAITEMENT SENTINEL-2
        img_data[img_data == -10000] = np.nan
        img_data /= 10_000.0

        if img_data.shape[0] < 3:
            rgb = np.stack([img_data[0], img_data[0], img_data[0]], axis=-1)
        else:
            rgb = np.stack([img_data[2], img_data[1], img_data[0]], axis=-1)

        rgb = np.nan_to_num(rgb, nan=0.0)
        rgb = np.clip(rgb / 0.3, 0, 1)
        final_img_uint8 = (rgb * 255).astype(np.uint8)

    else:
        # TRAITEMENT BRUT
        img_data = np.nan_to_num(img_data, nan=0.0)

        if img_data.shape[0] == 1:
            # Image mono-bande
            out_img = img_data[0]
        else:
            # Si multi-bandes-> prend les 3 premières
            out_img = np.stack([img_data[0], img_data[1], img_data[2]], axis=-1)

        # Pour qu'un masque [0, 1] soit visible en PNG, on le met à l'échelle [0, 255].
        if out_img.max() <= 1.0 and out_img.max() > 0:
            final_img_uint8 = (out_img * 255).astype(np.uint8)
        else:
            final_img_uint8 = np.clip(out_img, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.fromarray(final_img_uint8)
    img.save(out_path)
    print(f" -> Sauvegardé ({'Brut' if is_raw_mask else 'Normalisé'}) : {out_path}")

def get_flat_filename(file_path, base_dir, category):
    """
    Construit le nom de fichier plat avec les préfixes et suffixes appropriés.
    """
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    base_name = os.path.splitext(parts[-1])[0]

    parent_dir = parts[-2] if len(parts) >= 2 else ""

    doy_prefix = base_name

    if category == "predictions":
        # conserve l'intégralité du nom pour éviter les écrasements
        if "ref" in base_name.lower():
            suffix = "_ref"
            if doy_prefix.lower().endswith("_ref"):
                doy_prefix = doy_prefix[:-4]
        elif "pred" in base_name.lower():
            suffix = "_pred"
            if doy_prefix.lower().endswith("_pred"):
                doy_prefix = doy_prefix[:-5]
        else:
            suffix = ""

        return f"{doy_prefix}{suffix}.png"

    elif category == "output_unet":
        # type (prediction/temporal_difference) est déduit depuis le dossier parent
        if "pred" in parent_dir.lower():
            suffix = "_prediction_unet"
        elif "diff" in parent_dir.lower() or "temporal_difference" in parent_dir.lower():
            suffix = "_temporal_difference_unet"
        else:
            # Fallback si le parent n'est pas explicite
            if "pred" in base_name.lower():
                suffix = "_prediction_unet"
            elif "diff" in base_name.lower():
                suffix = "_temporal_difference_unet"
            else:
                suffix = "_unet"

        #  cas où le base_name contiendrait déjà _pred ou _diff à la fin
        for word in ["_pred", "_diff"]:
            if doy_prefix.lower().endswith(word):
                doy_prefix = doy_prefix[:-len(word)]

        return f"{doy_prefix}{suffix}.png"

    elif category == "change_detection":
        # le DOY exclusif est le nom du dossier parent
        if "huv_final_cube" in base_name.lower():
            doy_prefix = parent_dir
            suffix = "_kervrann"
        elif "kervrann" in base_name.lower():
            doy_prefix = base_name.lower().replace("_kervrann", "").replace("kervrann", "")
            suffix = "_kervrann"
        else:
            suffix = "_cd"

        return f"{doy_prefix}{suffix}.png"

    else:
        return f"{base_name}.png"

def main():
    parser = argparse.ArgumentParser(description="Convert TIFFs to PNGs into a flat directory.")
    parser.add_argument("base_path", type=str, help="Base path (e.g., forecasting/30SWH_24_c1_g1)")
    args = parser.parse_args()

    base_dir = os.path.normpath(args.base_path)
    output_dir = os.path.join(base_dir, "png_images")

    if not os.path.exists(base_dir):
        print(f"Erreur : Le dossier source '{base_dir}' n'existe pas.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dossier source : {base_dir}")
    print(f"Dossier de sortie (Flat) : {output_dir}")

    # PREDICTIONS
    pred_dir = os.path.join(base_dir, "predictions")
    if os.path.exists(pred_dir):
        print(f"\nTraitement des prédictions {pred_dir}")
        tif_files = glob.glob(os.path.join(pred_dir, "**", "*.tif"), recursive=True)
        for file_path in tif_files:
            print(file_path)
            new_filename = get_flat_filename(file_path, base_dir, "predictions")
            out_path = os.path.join(output_dir, new_filename)
            convert_tif_to_png(file_path, out_path, is_raw_mask=False)

    # CHANGE DETECTION: kervrann
    cd_dir = os.path.join(base_dir, "change_detection")
    if os.path.exists(cd_dir):
        print(f"\nTraitement de kervrann {cd_dir}")
        huv_files = glob.glob(os.path.join(cd_dir, "**", "huv_final_cube.tif"), recursive=True)
        for file_path in huv_files:
            new_filename = get_flat_filename(file_path, base_dir, "change_detection")
            out_path = os.path.join(output_dir, new_filename)
            convert_tif_to_png(file_path, out_path, is_raw_mask=True)

    # CHANGE DETECTION: UNET
    unet_dir = os.path.join(base_dir, "output_unet")
    if os.path.exists(unet_dir):
        print(f"\nTraitement du UNET {unet_dir}")
        # Traitement des PNG existants
        png_files = glob.glob(os.path.join(unet_dir, "**", "*.png"), recursive=True)
        for file_path in png_files:
            new_filename = get_flat_filename(file_path, base_dir, "output_unet")
            out_path = os.path.join(output_dir, new_filename)
            shutil.copy2(file_path, out_path)
            print(f" -> Copié (Existant) : {out_path}")

        # Traitement des TIF
        tif_files = glob.glob(os.path.join(unet_dir, "**", "*.tif"), recursive=True)
        for file_path in tif_files:
            new_filename = get_flat_filename(file_path, base_dir, "output_unet")
            out_path = os.path.join(output_dir, new_filename)
            convert_tif_to_png(file_path, out_path, is_raw_mask=False)

    print("\nDONE")

if __name__ == "__main__":
    main()