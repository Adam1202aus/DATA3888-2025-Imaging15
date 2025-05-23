# src/ml/rf/train.py
import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import time
import sys


def load_hog_data_for_selected_folds(
        base_feature_dir: Path,
        colorspace: str,
        fold_id_stems_to_load: list[str]
):
    all_X_arrays = []
    all_y_lists = []

    hog_colorspace_path = base_feature_dir / f"hog_{colorspace}"

    if not fold_id_stems_to_load:
        print(
            f"  Warning: No fold ID stems provided to load_hog_data_for_selected_folds for colorspace {colorspace}. Returning empty.")
        return np.array([]), []

    print(f"  Attempting to load HOG data from stems: {fold_id_stems_to_load} in {hog_colorspace_path}")
    for fold_id_stem in fold_id_stems_to_load:
        feature_file = hog_colorspace_path / f"{fold_id_stem}_hog_features.pkl"
        label_file = hog_colorspace_path / f"{fold_id_stem}_hog_labels.pkl"

        if feature_file.exists() and label_file.exists():
            try:
                X_fold = joblib.load(feature_file)
                y_fold = joblib.load(label_file)

                if not isinstance(X_fold, np.ndarray) or X_fold.size == 0:
                    print(
                        f"  Warning: HOG feature file {feature_file} for stem '{fold_id_stem}' is empty or not a NumPy array. Skipping.")
                    continue
                if not isinstance(y_fold, list) or len(y_fold) == 0:
                    print(
                        f"  Warning: HOG label file {label_file} for stem '{fold_id_stem}' is empty or not a list. Skipping.")
                    continue
                if X_fold.shape[0] != len(y_fold):
                    print(f"  Warning: Mismatch in number of samples between features ({X_fold.shape[0]}) "
                          f"and labels ({len(y_fold)}) for stem '{fold_id_stem}'. Skipping.")
                    continue

                all_X_arrays.append(X_fold)
                all_y_lists.extend(y_fold)
                print(f"    Successfully loaded {X_fold.shape[0]} samples from stem '{fold_id_stem}'.")
            except Exception as e:
                print(
                    f"  Error loading HOG data for stem '{fold_id_stem}' from {feature_file}/{label_file}: {e}. Skipping.")
        else:
            print(f"  Warning: HOG data for stem '{fold_id_stem}' not found. Searched for:")
            print(f"    Feature file: {feature_file} (Exists: {feature_file.exists()})")
            print(f"    Label file: {label_file} (Exists: {label_file.exists()})")
            print(f"  Skipping this fold's data.")

    if not all_X_arrays:
        print(f"  No HOG data successfully loaded for any of the provided stems in {hog_colorspace_path}.")
        return np.array([]), []

    final_X = np.vstack(all_X_arrays)
    final_y = all_y_lists

    print(f"  Total HOG samples loaded: {final_X.shape[0]}, Total labels: {len(final_y)}")
    return final_X, final_y


def train_rf_single_fixed_split(
        output_dir: Path,
        base_feature_dir: Path,
        colorspace: str,
        val_fold_hog_stem: str,
        all_hog_stems_json: str,
        n_estimators: int,
        max_depth: int = None,
        random_state: int = 42,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== Training RandomForest (Fixed Split: Validate on HOG stem '{val_fold_hog_stem}') =====")
    print(f"  Output directory: {output_dir}")
    print(f"  Base HOG feature dir: {base_feature_dir}")
    print(f"  HOG Colorspace: {colorspace}")

    try:
        all_available_hog_stems = json.loads(all_hog_stems_json)
        if not isinstance(all_available_hog_stems, list):
            raise ValueError("Decoded JSON for all_hog_stems_json is not a list.")
    except json.JSONDecodeError:
        print(f"FATAL: Error decoding JSON from all_hog_stems_json: {all_hog_stems_json}")
        sys.exit(1)
    except ValueError as e:
        print(f"FATAL: Invalid format for all_hog_stems_json: {e}")
        sys.exit(1)

    print(f"  All available HOG stems for this run: {all_available_hog_stems}")
    print(f"  Validation HOG stem: {val_fold_hog_stem}")

    train_hog_stems = [stem for stem in all_available_hog_stems if stem != val_fold_hog_stem]

    if not train_hog_stems:
        print(
            f"  ERROR: No training HOG stems found (all_stems: {all_available_hog_stems}, val_stem: {val_fold_hog_stem}). "
            "This might happen if only one HOG stem is provided or val_stem is not in all_stems.")
        (output_dir / "error.txt").write_text("No training HOG data stems available.")
        sys.exit(1)

    print(f"  Loading TRAINING HOG data from stems: {train_hog_stems}")
    X_train, y_train = load_hog_data_for_selected_folds(base_feature_dir, colorspace, train_hog_stems)

    print(f"  Loading VALIDATION HOG data from stem: {val_fold_hog_stem}")
    X_val, y_val = load_hog_data_for_selected_folds(base_feature_dir, colorspace, [val_fold_hog_stem])

    if X_train.size == 0 or not y_train:
        print(f"  ERROR: Training HOG data is empty after loading. Cannot proceed.")
        (output_dir / "error.txt").write_text("Training HOG data empty.")
        sys.exit(1)
    if X_val.size == 0 or not y_val:
        print(f"  ERROR: Validation HOG data is empty after loading. Cannot proceed.")
        (output_dir / "error.txt").write_text("Validation HOG data empty.")
        sys.exit(1)

    print(f"  RF Training on {X_train.shape[0]} HOG samples, Validating on {X_val.shape[0]} HOG samples.")
    print(f"  Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_duration = end_time - start_time
    (output_dir / "training_time.txt").write_text(f"{training_duration:.2f} seconds")
    print(f"  RF Training completed in {training_duration:.2f} seconds.")

    model_save_path = output_dir / "model.pkl"
    joblib.dump(model, model_save_path)
    print(f"  Model saved to {model_save_path}")

    y_pred_val = model.predict(X_val)

    present_labels_numeric = sorted(list(np.unique(np.concatenate((y_val, y_pred_val)))))
    present_labels_str = [str(x) for x in present_labels_numeric]

    report_str = classification_report(y_val, y_pred_val, labels=present_labels_numeric,
                                       target_names=present_labels_str, digits=4, zero_division=0)
    report_dict = classification_report(y_val, y_pred_val, labels=present_labels_numeric,
                                        target_names=present_labels_str, digits=4, output_dict=True,
                                        zero_division=0)

    print(f"\n  Classification Report (on validation HOG set '{val_fold_hog_stem}'):\n{report_str}")
    (output_dir / "report.txt").write_text(report_str)
    with open(output_dir / "report_dict.json", "w", encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4)

    np.save(str(output_dir / "y_true_val.npy"), np.array(y_val))
    np.save(str(output_dir / "y_pred_val.npy"), y_pred_val)

    cm = confusion_matrix(y_val, y_pred_val, labels=present_labels_numeric)
    (output_dir / "confusion_matrix_labels.json").write_text(json.dumps(present_labels_str), encoding='utf-8')
    np.savetxt(str(output_dir / "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    print(
        f"  [OK] RF training for HOG colorspace '{colorspace}' (Val HOG: '{val_fold_hog_stem}') finished. Results in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest model for a fixed split using HOG features.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model and reports for this run.")
    parser.add_argument("--base_feature_dir", type=str, required=True,
                        help="Base directory where HOG features for all folds are stored (e.g. data/features_ml).")
    parser.add_argument("--colorspace", type=str, required=True,
                        help="Colorspace of the HOG features (e.g. original). Used to find hog_<colorspace> subdir.")

    parser.add_argument("--val_fold_hog_stem", type=str, required=True,
                        help="The ID stem of the HOG fold file to be used as validation data (e.g. fold_5 or fold_5_sampled_0.5).")
    parser.add_argument("--all_hog_stems_json", type=str, required=True,
                        help="JSON string list of all HOG fold ID stems relevant for this k-fold setup "
                             "(e.g. '[\"fold_1_s\", \"fold_2_s\", ..., \"fold_5_s\"]'). These stems are used to load HOG features.")

    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.")
    parser.add_argument("--min_samples_split", type=int, default=2,
                        help="Minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1,
                        help="Minimum number of samples required to be at a leaf node.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")

    args = parser.parse_args()

    train_rf_single_fixed_split(
        output_dir=Path(args.output_dir),
        base_feature_dir=Path(args.base_feature_dir),
        colorspace=args.colorspace,
        val_fold_hog_stem=args.val_fold_hog_stem,
        all_hog_stems_json=args.all_hog_stems_json,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth != 0 else None,
        random_state=args.random_state,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )