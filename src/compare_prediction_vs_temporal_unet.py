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
        img_data = np.nan_to_num(img_data, nan=0.0)

        if img_data.shape[0] == 1:
            # Image mono-bande
            out_img = img_data[0]
        else:
            # Si multi-bandes, on prend les 3 premières
            out_img = np.stack([img_data[0], img_data[1], img_data[2]], axis=-1)

        # Pour qu'un masque [0, 1] soit visible en PNG, on le met à l'échelle [0, 255].
        if out_img.max() <= 1.0 and out_img.max() > 0:
            final_img_uint8 = (out_img * 255).astype(np.uint8)
        else:
            final_img_uint8 = np.clip(out_img, 0, 255).astype(np.uint8)

    # Sauvegarde
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.fromarray(final_img_uint8)
    img.save(out_path)
    print(f" -> Sauvegardé ({'Brut' if is_raw_mask else 'Normalisé'}) : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert TIFFs to PNGs.")
    parser.add_argument("base_path", type=str, help="Base path (e.g., forecasting/30SWH_24_c1_g1)")
    args = parser.parse_args()

    base_dir = os.path.normpath(args.base_path)
    output_dir = os.path.join(base_dir, "png_images")

    if not os.path.exists(base_dir):
        print(f"Erreur : Le dossier source '{base_dir}' n'existe pas.")
        return

    print(f"Dossier source : {base_dir}")
    print(f"Dossier de sortie : {output_dir}")

    # PREDICTIONS (Normalisation Sentinel-2)
    pred_dir = os.path.join(base_dir, "predictions")
    if os.path.exists(pred_dir):
        print("\nTraitement des prédictions (valeurs normalisées)")
        tif_files = glob.glob(os.path.join(pred_dir, "**", "*.tif"), recursive=True)
        for file_path in tif_files:
            rel_path = os.path.relpath(file_path, base_dir)
            out_path = os.path.join(output_dir, rel_path).replace(".tif", ".png")
            convert_tif_to_png(file_path, out_path, is_raw_mask=False)

    # CHANGE DETECTION (Valeurs brutes)
    cd_dir = os.path.join(base_dir, "change_detection")
    if os.path.exists(cd_dir):
        print("\nTraitement de change_detection (valeurs brutes)")
        huv_files = glob.glob(os.path.join(cd_dir, "**", "huv_final_cube.tif"), recursive=True)
        for file_path in huv_files:
            rel_path = os.path.relpath(file_path, base_dir)
            out_path = os.path.join(output_dir, rel_path).replace(".tif", ".png")
            convert_tif_to_png(file_path, out_path, is_raw_mask=True)

    # OUTPUT UNET (Valeurs brutes & Copies)
    unet_dir = os.path.join(base_dir, "output_unet")
    if os.path.exists(unet_dir):
        print("\nTraitement de output_unet (valeurs brutes)")
        png_files = glob.glob(os.path.join(unet_dir, "**", "*.png"), recursive=True)
        for file_path in png_files:
            rel_path = os.path.relpath(file_path, base_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(file_path, out_path)
            print(f" -> Copié (Existant) : {out_path}")

        tif_files = glob.glob(os.path.join(unet_dir, "**", "*.tif"), recursive=True)
        for file_path in tif_files:
            rel_path = os.path.relpath(file_path, base_dir)
            out_path = os.path.join(output_dir, rel_path).replace(".tif", ".png")
            # Changed to True here!
            convert_tif_to_png(file_path, out_path, is_raw_mask=True)

    print("DONE")

if __name__ == "__main__":
    main()
