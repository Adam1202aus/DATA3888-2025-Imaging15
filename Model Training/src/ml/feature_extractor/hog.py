# src/ml/feature_extractor/hog.py
from skimage.feature import hog as skimage_hog
from skimage.transform import resize as skimage_resize
from skimage.color import (
    rgb2gray as skimage_rgb2gray_convert,
    rgb2ycbcr as skimage_rgb2ycbcr_convert,
    rgb2lab as skimage_rgb2lab_convert,
    rgb2hed as skimage_rgb2hed_convert,
    hed2rgb
)
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
from skimage.io import imsave

import numpy as np
import joblib
import argparse
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import pandas as pd
import sys

DEBUG_IMAGE_COUNT = 3
DEBUG_IMAGES_SAVED = {}


def save_debug_image(image_array: np.ndarray, original_stem: str, target_colorspace: str, channel_name: str = ""):
    global DEBUG_IMAGES_SAVED

    cs_key = f"{target_colorspace}_{channel_name}" if channel_name else target_colorspace
    if cs_key not in DEBUG_IMAGES_SAVED:
        DEBUG_IMAGES_SAVED[cs_key] = 0

    if DEBUG_IMAGES_SAVED[cs_key] < DEBUG_IMAGE_COUNT:
        try:
            debug_output_dir = Path(f"./debug_hog_inputs/{target_colorspace}")
            debug_output_dir.mkdir(parents=True, exist_ok=True)

            filename_suffix = f"_{channel_name}" if channel_name else ""
            save_path = debug_output_dir / f"{original_stem}_{target_colorspace}{filename_suffix}.png"


            if image_array.ndim == 2:
                img_to_save = img_as_ubyte(rescale_intensity(image_array, out_range=(0, 1)))
            elif image_array.ndim == 3 and image_array.shape[2] == 3:
                ch1_norm = rescale_intensity(image_array[:, :, 0], out_range=(0, 1))
                ch2_norm = rescale_intensity(image_array[:, :, 1], out_range=(0, 1))
                ch3_norm = rescale_intensity(image_array[:, :, 2], out_range=(0, 1))
                img_to_save = img_as_ubyte(np.dstack((ch1_norm, ch2_norm, ch3_norm)))
            else:
                print(f"  Debug Save: Skipping unsupported image shape {image_array.shape} for {save_path}")
                return

            imsave(save_path, img_to_save)
            DEBUG_IMAGES_SAVED[cs_key] += 1
        except Exception as e:
            print(f"  DEBUG: Error saving debug image {original_stem} for {target_colorspace}: {e}")


def extract_hog_features(image_path: Path,
                         target_colorspace: str = "original",
                         target_size=(128, 128),
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         orientations=9,
                         save_debug_flag=False):
    try:
        pil_img_rgb = PIL.Image.open(image_path)
        img_array_rgb_original_pil_mode = pil_img_rgb.mode
        img_array_rgb = np.array(pil_img_rgb)

        if img_array_rgb.ndim == 2:
            img_array_rgb_compatible = skimage_rgb2gray_convert(img_array_rgb)
            if target_colorspace not in ["grayscale", "original"]:
                print(
                    f"  Info: Original image {image_path.name} (mode: {img_array_rgb_original_pil_mode}) is grayscale. "
                    f"Forcing target_colorspace to 'grayscale' for HOG.")
                target_colorspace = "grayscale"
        elif img_array_rgb.ndim == 3 and img_array_rgb.shape[2] == 1:
            img_array_rgb_compatible = img_array_rgb[:, :, 0]
            if target_colorspace not in ["grayscale", "original"]:
                print(
                    f"  Info: Original image {image_path.name} (shape: {img_array_rgb.shape}) is effectively grayscale. "
                    f"Forcing target_colorspace to 'grayscale' for HOG.")
                target_colorspace = "grayscale"
        elif img_array_rgb.ndim == 3 and img_array_rgb.shape[2] == 4:
            img_array_rgb_compatible = img_array_rgb[:, :, :3]
        elif img_array_rgb.ndim == 3 and img_array_rgb.shape[2] == 3:
            img_array_rgb_compatible = img_array_rgb
        else:
            print(f"  Error: Unhandled original image format for {image_path.name} "
                  f"(mode: {img_array_rgb_original_pil_mode}, shape: {img_array_rgb.shape}). Attempting grayscale conversion of original.")
            try:
                img_array_rgb_compatible = skimage_rgb2gray_convert(img_array_rgb)
                target_colorspace = "grayscale"
            except Exception as conv_e:
                print(f"    Failed to convert to grayscale: {conv_e}. Skipping image.")
                return None


        img_float_rgb = img_array_rgb_compatible.astype(np.float32)
        if np.max(img_float_rgb) > 1.01:
            img_float_rgb = img_float_rgb / 255.0

        img_to_process = None
        if target_colorspace == "grayscale":
            img_to_process = skimage_rgb2gray_convert(img_float_rgb)
        elif target_colorspace == "ycbcr":
            img_to_process = skimage_rgb2ycbcr_convert(img_float_rgb)
        elif target_colorspace == "cielab":
            img_to_process = skimage_rgb2lab_convert(
                img_float_rgb)
        elif target_colorspace == "hed":
            img_to_process = skimage_rgb2hed_convert(img_float_rgb)
        elif target_colorspace == "original":
            img_to_process = img_float_rgb.copy()
        else:
            print(f"  Warning: Unknown target_colorspace '{target_colorspace}'. Using original RGB for HOG.")
            img_to_process = img_float_rgb.copy()

        if save_debug_flag:
            original_stem = image_path.stem
            if target_colorspace == "hed" and img_to_process.ndim == 3:
                null_channel = np.zeros_like(img_to_process[:, :, 0])
                h_stain_rgb = hed2rgb(np.dstack((img_to_process[:, :, 0], null_channel, null_channel)))
                e_stain_rgb = hed2rgb(np.dstack((null_channel, img_to_process[:, :, 1], null_channel)))
                d_stain_rgb = hed2rgb(np.dstack((null_channel, null_channel, img_to_process[:, :, 2])))
                save_debug_image(h_stain_rgb, original_stem, target_colorspace, "Hema")
                save_debug_image(e_stain_rgb, original_stem, target_colorspace, "Eos")
                save_debug_image(d_stain_rgb, original_stem, target_colorspace, "DAB")
            elif img_to_process.ndim == 2:
                save_debug_image(img_to_process, original_stem, target_colorspace)
            elif img_to_process.ndim == 3 and img_to_process.shape[2] == 3:
                if target_colorspace in ["ycbcr", "cielab"]:
                    save_debug_image(img_to_process[:, :, 0], original_stem, target_colorspace, "Ch1")
                    save_debug_image(img_to_process[:, :, 1], original_stem, target_colorspace, "Ch2")
                    save_debug_image(img_to_process[:, :, 2], original_stem, target_colorspace, "Ch3")
                else:
                    save_debug_image(img_to_process, original_stem, target_colorspace)

        if img_to_process.ndim == 2:
            channel_resized = skimage_resize(img_to_process, target_size,
                                             anti_aliasing=True)
            hog_features = skimage_hog(channel_resized, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                       cells_per_block=cells_per_block, feature_vector=True,
                                       channel_axis=None)
            return hog_features

        elif img_to_process.ndim == 3 and img_to_process.shape[2] > 0:
            resized_multichannel_img = skimage_resize(img_to_process, (*target_size, img_to_process.shape[2]),
                                                      anti_aliasing=True)

            hog_features = skimage_hog(resized_multichannel_img, orientations=orientations,
                                       pixels_per_cell=pixels_per_cell,
                                       cells_per_block=cells_per_block, feature_vector=True,
                                       channel_axis=-1)
            return hog_features
        else:
            print(f"  Warning: Image {image_path.name} (target: {target_colorspace}, shape: {img_to_process.shape}) "
                  f"not suitable for HOG. Attempting grayscale HOG of original RGB.")
            img_gray_original = skimage_rgb2gray_convert(img_float_rgb)
            img_resized = skimage_resize(img_gray_original, target_size, anti_aliasing=True)
            return skimage_hog(img_resized, orientations=orientations, pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block, feature_vector=True, channel_axis=None)

    except FileNotFoundError:
        print(f"  Error: Image file not found at {image_path}. Skipping.")
        return None
    except Exception as e:
        print(
            f"  Error extracting HOG for {image_path.name} (target_colorspace: {target_colorspace}): {e}. Returning None.")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract HOG features for a specific colorspace and fold CSV, converting from original RGB.")
    parser.add_argument('--image_data_root', type=str, required=True,
                        help="Base directory for ORIGINAL RGB image data (e.g., data/interim/original).")
    parser.add_argument('--target_colorspace', type=str, default="original",
                        choices=["original", "grayscale", "ycbcr", "cielab", "hed"],
                        help="Target colorspace to convert to before HOG extraction.")
    parser.add_argument('--splits_dir', type=str, required=True,
                        help="Directory containing the split CSV file (listing original images).")
    parser.add_argument('--fold_id', type=str, required=True,
                        help="Fold identifier stem of the CSV (e.g., 'fold_1'). This CSV lists original images.")
    parser.add_argument('--output_dir_base', type=str, required=True,
                        help="Base directory to save HOG features. Output will be in output_dir_base/hog_<target_colorspace>/")
    parser.add_argument('--target_image_height', type=int, default=128)
    parser.add_argument('--target_image_width', type=int, default=128)
    parser.add_argument('--pixels_per_cell_h', type=int, default=8)
    parser.add_argument('--pixels_per_cell_w', type=int, default=8)
    parser.add_argument('--cells_per_block_h', type=int, default=2)
    parser.add_argument('--cells_per_block_w', type=int, default=2)
    parser.add_argument('--orientations', type=int, default=9)
    parser.add_argument('--debug_save_converted_images', action='store_true',
                        help="If set, save a few converted images for each colorspace to 'debug_hog_inputs/' for verification.")

    args = parser.parse_args()

    source_image_data_root = Path(args.image_data_root)
    fold_csv_path = Path(args.splits_dir) / f"{args.fold_id}.csv"
    output_dir_for_hog_variant = Path(args.output_dir_base) / f"hog_{args.target_colorspace}"
    output_dir_for_hog_variant.mkdir(parents=True, exist_ok=True)

    target_size_tuple = (args.target_image_height, args.target_image_width)
    pixels_per_cell_tuple = (args.pixels_per_cell_h, args.pixels_per_cell_w)
    cells_per_block_tuple = (args.cells_per_block_h, args.cells_per_block_w)

    global DEBUG_IMAGES_SAVED
    DEBUG_IMAGES_SAVED = {}

    if not source_image_data_root.is_dir():
        print(f"ERROR: Source image data root '{source_image_data_root}' not found or not a directory.")
        sys.exit(1)
    if not fold_csv_path.exists():
        print(f"ERROR: Split CSV file not found: {fold_csv_path}. Cannot extract HOG features.")
        sys.exit(1)

    try:
        df_split = pd.read_csv(fold_csv_path)
    except pd.errors.EmptyDataError:
        df_split = pd.DataFrame()

    if df_split.empty:
        print(
            f"Split CSV {fold_csv_path} is empty. Creating empty placeholder HOG files for target colorspace '{args.target_colorspace}'.")
        return

    features_list = []
    labels_list = []
    image_paths_processed = []

    print(
        f"Extracting HOG for target colorspace '{args.target_colorspace}', using original images listed in '{fold_csv_path.name}' ({len(df_split)} entries)")
    if args.debug_save_converted_images:
        print(
            f"  DEBUG: Will save up to {DEBUG_IMAGE_COUNT} sample converted images per type to ./debug_hog_inputs/{args.target_colorspace}/")

    for index, row in tqdm(df_split.iterrows(), total=df_split.shape[0],
                           desc=f"HOG (TargetCS: {args.target_colorspace}, CSV: {args.fold_id})"):
        relative_image_path_str = row['rel_path']
        full_image_path = source_image_data_root / Path(relative_image_path_str)

        if not full_image_path.exists():
            print(f"  Warning: Original image not found at {full_image_path}. Skipping.")
            continue

        hog_feature = extract_hog_features(
            full_image_path,
            target_colorspace=args.target_colorspace,
            target_size=target_size_tuple,
            pixels_per_cell=pixels_per_cell_tuple,
            cells_per_block=cells_per_block_tuple,
            orientations=args.orientations,
            save_debug_flag=args.debug_save_converted_images
        )
        if hog_feature is not None and hog_feature.size > 0:
            features_list.append(hog_feature)
            labels_list.append(row['label'])
            image_paths_processed.append(str(relative_image_path_str))
        else:
            print(
                f"  Warning: HOG feature extraction failed or returned empty for {full_image_path.name} (target: {args.target_colorspace}). Skipping this image.")

    if not features_list:
        print(
            f"Warning: No HOG features were successfully extracted for target colorspace '{args.target_colorspace}', split '{args.fold_id}'.")
        return

    X_features = np.vstack(features_list)
    base_output_filename_stem = args.fold_id

    feature_file_path = output_dir_for_hog_variant / f"{base_output_filename_stem}_hog_features.pkl"
    label_file_path = output_dir_for_hog_variant / f"{base_output_filename_stem}_hog_labels.pkl"
    paths_file_path = output_dir_for_hog_variant / f"{base_output_filename_stem}_hog_paths.pkl"

    joblib.dump(X_features, feature_file_path)
    joblib.dump(labels_list, label_file_path)
    joblib.dump(image_paths_processed, paths_file_path)

    print(f"  Actual HOG feature file for '{args.target_colorspace}' saved: {feature_file_path}")
    print(f"  Actual label file for '{args.target_colorspace}' saved:   {label_file_path}")
    print(f"  Actual paths file for '{args.target_colorspace}' saved:   {paths_file_path}")
    print(
        f"[OK] HOG features for target colorspace '{args.target_colorspace}' (from CSV '{args.fold_id}') saved to {output_dir_for_hog_variant}")


if __name__ == '__main__':
    main()