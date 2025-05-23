# src/common/make_splits.py
import argparse
import csv
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
import numpy as np


def get_sample_id_from_path(path: Path) -> str:
    stem = path.stem
    if '_aug' in stem:
        return stem.split('_aug')[0]
    return stem


def collect_data_from_structure(image_data_root: Path):
    samples = []
    labels = []
    sample_ids = []

    train_subdir = image_data_root / "train"
    if not train_subdir.exists() or not train_subdir.is_dir():
        raise FileNotFoundError(
            f"'train' subdirectory not found in {image_data_root}. "
            "Splits are typically made from the training set."
        )

    print(f"Collecting image data from: {train_subdir}")
    class_counts = defaultdict(int)
    for class_dir in sorted(train_subdir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        image_count_for_class = 0
        for ext_pattern in ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]:
            for img_file_path in class_dir.glob(ext_pattern):
                samples.append(img_file_path)
                labels.append(class_name)
                sample_ids.append(get_sample_id_from_path(img_file_path))
                image_count_for_class += 1
        if image_count_for_class > 0:
            class_counts[class_name] = image_count_for_class
        else:
            print(f"Warning: No images found in class directory: {class_dir}")

    if not samples:
        raise ValueError(f"No images collected from {train_subdir}. Check directory structure and image extensions.")

    print(f"Collected {len(samples)} image files from {len(class_counts)} classes.")
    for cls, count in class_counts.items():
        print(f"  Class '{cls}': {count} files")
    return samples, labels, sample_ids


def write_split_csv(output_path: Path, image_paths: list[Path], data_root_for_relative_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rel_path', 'label'])
        for p in image_paths:
            label = p.parent.name
            try:
                rel_path_obj = p.relative_to(data_root_for_relative_path)
                writer.writerow([rel_path_obj.as_posix(), label])
            except ValueError:
                print(
                    f"Warning: Could not make path {p} relative to {data_root_for_relative_path}. Using absolute path (not recommended).")
                writer.writerow([str(p), label])


def main():
    parser = argparse.ArgumentParser(description="Create Stratified K-Fold splits for image datasets.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to the root directory of the image data to split (e.g., data/interim/original). Should contain a 'train' subdirectory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the split CSV files (fold_X.csv) will be saved (e.g., data/splits).")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for Stratified K-Fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--sample_fraction", type=float, default=None,
                        help="Optional: Fraction of unique samples to use (0.0 to 1.0). Useful for creating smaller test splits.")
    args = parser.parse_args()

    data_root_path = Path(args.data_root)
    output_dir_path = Path(args.output_dir)

    all_image_paths, all_labels, all_unique_sample_ids = collect_data_from_structure(data_root_path)

    id_to_image_paths_map = defaultdict(list)
    for img_path, unique_id_val in zip(all_image_paths, all_unique_sample_ids):
        id_to_image_paths_map[unique_id_val].append(img_path)

    unique_ids_for_stratification = list(id_to_image_paths_map.keys())
    unique_id_labels = [all_labels[all_image_paths.index(id_to_image_paths_map[uid][0])] for uid in
                        unique_ids_for_stratification]

    print(f"Found {len(unique_ids_for_stratification)} unique sample IDs for stratification.")

    if args.sample_fraction is not None and 0.0 < args.sample_fraction <= 1.0:
        num_to_sample = int(len(unique_ids_for_stratification) * args.sample_fraction)
        if num_to_sample == 0 and len(unique_ids_for_stratification) > 0:
            num_to_sample = 1

        print(
            f"[INFO] Applying sample_fraction: {args.sample_fraction}. Aiming for {num_to_sample} unique samples for splits.")

        if num_to_sample < len(unique_ids_for_stratification) and num_to_sample > 0:
            try:
                full_indices = np.arange(len(unique_ids_for_stratification))
                sampled_indices, _ = train_test_split(
                    full_indices,
                    train_size=num_to_sample,
                    stratify=unique_id_labels,
                    random_state=args.seed
                )
                unique_ids_for_stratification = [unique_ids_for_stratification[i] for i in sampled_indices]
                unique_id_labels = [unique_id_labels[i] for i in sampled_indices]
                print(
                    f"Successfully selected {len(unique_ids_for_stratification)} unique samples after stratified sampling.")
            except ValueError as e:
                print(f"Warning: Stratified subsampling failed ({e}). Using random subsampling for unique IDs.")
                np.random.seed(args.seed)
                sampled_indices = np.random.choice(len(unique_ids_for_stratification), num_to_sample, replace=False)
                unique_ids_for_stratification = [unique_ids_for_stratification[i] for i in sampled_indices]
                unique_id_labels = [unique_id_labels[i] for i in sampled_indices]
        elif num_to_sample >= len(unique_ids_for_stratification):
            print(
                f"[INFO] sample_fraction ({args.sample_fraction}) results in using all {len(unique_ids_for_stratification)} unique samples.")

        if not unique_ids_for_stratification:
            print("Error: No samples selected after applying sample_fraction. Exiting.")
            return

    k_splits = args.k
    if len(unique_ids_for_stratification) < k_splits:
        print(
            f"Warning: Number of unique samples ({len(unique_ids_for_stratification)}) is less than k_splits ({k_splits}). Setting k_splits to {len(unique_ids_for_stratification)}.")
        k_splits = len(unique_ids_for_stratification)

    if k_splits < 2 and len(unique_ids_for_stratification) > 0:
        print(
            f"Warning: Not enough unique samples ({len(unique_ids_for_stratification)}) for {k_splits} folds. Cannot perform K-Fold split. Saving all as fold_1.")
        output_csv_path = output_dir_path / "fold_1.csv"
        all_img_paths_for_single_fold = []
        for uid in unique_ids_for_stratification:
            all_img_paths_for_single_fold.extend(id_to_image_paths_map[uid])
        write_split_csv(output_csv_path, all_img_paths_for_single_fold, data_root_path)
        print(
            f"[✓] All {len(all_img_paths_for_single_fold)} image samples written to {output_csv_path} as a single fold.")
        return

    print(f"Creating {k_splits} folds using {len(unique_ids_for_stratification)} unique samples for stratification...")
    skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=args.seed)

    for i, (train_idx_of_unique_ids, val_idx_of_unique_ids) in enumerate(
            skf.split(unique_ids_for_stratification, unique_id_labels), 1):
        val_unique_ids_for_this_fold = [unique_ids_for_stratification[idx] for idx in val_idx_of_unique_ids]
        validation_image_paths_for_this_fold = []
        for unique_id_val in val_unique_ids_for_this_fold:
            validation_image_paths_for_this_fold.extend(id_to_image_paths_map[unique_id_val])

        output_csv_path = output_dir_path / f"fold_{i}.csv"
        write_split_csv(output_csv_path, validation_image_paths_for_this_fold, data_root_path)
        print(
            f"[OK] Fold {i:>2}: {len(validation_image_paths_for_this_fold)} image files (from {len(val_unique_ids_for_this_fold)} unique IDs) written to {output_csv_path}")

    print(f"\nAll {k_splits} split files generated in {output_dir_path}")


if __name__ == "__main__":
    main()