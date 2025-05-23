# src/run_all_folds.py
import yaml
from pathlib import Path
import subprocess
import pandas as pd
import datetime
import shutil
import sys
import os
import argparse
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent

def load_config(config_path_from_root="config.yaml"):
    config_file = PROJECT_ROOT / config_path_from_root
    if not config_file.exists():
        print(f"FATAL ERROR: Main configuration file not found at {config_file}")
        sys.exit(1)
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_temporary_split_csv(original_split_csv_path: Path,
                               temp_target_dir: Path,
                               sample_fraction: float,
                               seed: int,
                               fold_num_for_log: int,
                               global_splits_dir_path: Path) -> Path:
    temp_target_dir.mkdir(parents=True, exist_ok=True)

    if not original_split_csv_path.exists():
        print(
            f"      Warning: Original split CSV {original_split_csv_path} for fold {fold_num_for_log} not found. Cannot create temp split.")
        temp_csv_path_missing = temp_target_dir / f"{original_split_csv_path.stem}_temp_missing_original.csv"
        pd.DataFrame(columns=['rel_path', 'label']).to_csv(temp_csv_path_missing, index=False, encoding='utf-8')
        return temp_csv_path_missing

    df_orig_split = pd.read_csv(original_split_csv_path)
    if df_orig_split.empty:
        print(f"      Warning: Original split CSV {original_split_csv_path} for fold {fold_num_for_log} is empty.")
        temp_csv_path_empty = temp_target_dir / f"{original_split_csv_path.stem}_temp_empty.csv"
        df_orig_split.head(0).to_csv(temp_csv_path_empty, index=False, encoding='utf-8')
        return temp_csv_path_empty

    num_to_sample_ideal = int(len(df_orig_split) * sample_fraction)
    if num_to_sample_ideal == 0 and len(df_orig_split) > 0:
        num_to_sample_ideal = 1

    df_test_split = pd.DataFrame()
    if 'label' in df_orig_split.columns and df_orig_split['label'].nunique() > 1 and len(df_orig_split) > df_orig_split[
        'label'].nunique() and num_to_sample_ideal > 0:
        try:
            min_samples_for_stratify = df_orig_split['label'].nunique()
            actual_sample_size = max(num_to_sample_ideal,
                                     min_samples_for_stratify if num_to_sample_ideal >= min_samples_for_stratify else 0)
            actual_sample_size = min(actual_sample_size, len(df_orig_split))

            if actual_sample_size < min_samples_for_stratify and actual_sample_size > 0:
                df_test_split = df_orig_split.sample(n=actual_sample_size, random_state=seed, replace=False if len(
                    df_orig_split) >= actual_sample_size else True)
            elif actual_sample_size > 0:
                df_test_split, _ = train_test_split(df_orig_split, train_size=actual_sample_size,
                                                    stratify=df_orig_split['label'], random_state=seed)

            if df_test_split.empty and not df_orig_split.empty and num_to_sample_ideal > 0:
                df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed)
        except ValueError as e:
            df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed,
                                                 replace=False if len(df_orig_split) >= min(num_to_sample_ideal,
                                                                                            len(df_orig_split)) else True)
    elif not df_orig_split.empty and num_to_sample_ideal > 0:
        df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed,
                                             replace=False if len(df_orig_split) >= min(num_to_sample_ideal,
                                                                                        len(df_orig_split)) else True)

    if df_test_split.empty and not df_orig_split.empty and num_to_sample_ideal > 0:
        df_test_split = df_orig_split.sample(n=1, random_state=seed)

    temp_csv_path = temp_target_dir / f"{original_split_csv_path.stem}_samplefrac_{sample_fraction:.3f}.csv"
    df_test_split.to_csv(temp_csv_path, index=False, encoding='utf-8')
    return temp_csv_path


def run_command(command_list: list, step_name: str, error_on_fail=True, timeout_seconds=7200, pbar: tqdm = None):
    str_command_list = [str(c) for c in command_list]
    original_postfix = ""
    if pbar and hasattr(pbar, 'postfix') and pbar.postfix is not None: original_postfix = str(pbar.postfix)
    current_action_postfix = f"Running: {step_name[:30]}..."
    if pbar:
        pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {current_action_postfix}")
    else:
        print(f"--- Starting: {step_name} ---\n      Command: {' '.join(str_command_list)}")
    process = None
    try:
        process = subprocess.Popen(str_command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True)
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line: tqdm.write(f"      [LOG - {step_name[:20]}]: {line}", file=sys.stdout)
        process.wait(timeout=timeout_seconds)
        stderr_output = process.stderr.read() if process.stderr else ""
        if process.returncode != 0:
            final_postfix_msg = f"ERROR: {step_name[:30]}"
            if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {final_postfix_msg}")
            print(f"ERROR during {step_name} (Code: {process.returncode}): {' '.join(str_command_list)}")
            if stderr_output.strip(): print(f"      Stderr:\n{stderr_output.strip()}")
            if error_on_fail: raise subprocess.CalledProcessError(process.returncode, str_command_list,
                                                                  stderr=stderr_output)
            return False
        else:
            final_postfix_msg = f"Finished: {step_name[:30]}"
            if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {final_postfix_msg}")
            if stderr_output.strip(): tqdm.write(
                f"      [STDERR - {step_name[:20]} - Non-fatal]:\n{stderr_output.strip()}", file=sys.stderr)
            return True
    except subprocess.TimeoutExpired:
        final_postfix_msg = f"TIMEOUT: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {final_postfix_msg}")
        print(f"TIMEOUT ERROR: {step_name} exceeded {timeout_seconds}s. Cmd: {' '.join(str_command_list)}")
        if process: process.kill(); process.communicate(timeout=5)
        if error_on_fail: raise
        return False
    except FileNotFoundError as e:
        final_postfix_msg = f"NOT FOUND: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {final_postfix_msg}")
        print(f"ERROR: Command or script not found for {step_name}. Cmd: {' '.join(str_command_list)}. Details: {e}")
        if error_on_fail: raise
        return False
    except Exception as e:
        final_postfix_msg = f"EXCEPTION: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}, {final_postfix_msg}")
        print(f"AN UNEXPECTED ERROR OCCURRED during {step_name}: {e}")
        if process: process.kill(); process.communicate(timeout=5)
        if error_on_fail: raise
        return False
    finally:
        if pbar: pbar.set_postfix_str(f"{original_postfix.split(', Running:')[0].strip()}")


def main(run_mode: str, requested_folds: list[int] | None):
    cfg = load_config()
    python_executable = sys.executable
    is_test_run = (run_mode == "test")

    print(f"--- Starting Experiment Orchestration (Mode: {run_mode.upper()}) ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python Executable: {python_executable}")

    k_folds_total_from_config = cfg['cv'].get('k_folds', 5)

    if requested_folds:
        folds_to_process_numbers = []
        for fold_num in requested_folds:
            if 1 <= fold_num <= k_folds_total_from_config:
                if fold_num not in folds_to_process_numbers:
                    folds_to_process_numbers.append(fold_num)
            else:
                print(
                    f"      Warning: Specified fold {fold_num} is out of range (1-{k_folds_total_from_config}). It will be skipped.")

        if not folds_to_process_numbers:
            print(
                f"      Error: No valid folds found in --run_folds input {requested_folds}. Defaulting to run all {k_folds_total_from_config} folds.")
            folds_to_process_numbers = list(range(1, k_folds_total_from_config + 1))
        else:
            folds_to_process_numbers.sort()
            print(
                f"--- Processing specified validation folds: {folds_to_process_numbers} (out of {k_folds_total_from_config} total folds) ---")

    else:
        folds_to_process_numbers = list(range(1, k_folds_total_from_config + 1))
        print(
            f"--- No specific folds provided via --run_folds. Running all {len(folds_to_process_numbers)} folds for cross-validation (1 to {k_folds_total_from_config}) ---")

    current_run_timeout = cfg.get('subprocess_timeout_seconds', 36000)
    if is_test_run and cfg.get('test_run', {}).get('subprocess_timeout_seconds_override') is not None:
        current_run_timeout = cfg['test_run']['subprocess_timeout_seconds_override']
    print(f"Using {'TEST MODE' if is_test_run else 'FULL MODE'} timeout: {current_run_timeout} seconds")

    active_hyperparameters = cfg['hyperparameters'].copy()
    sample_fraction_for_test_run = 1.0

    if is_test_run:
        sample_fraction_for_test_run = cfg.get('test_run', {}).get('sample_fraction', 0.1)
        print_test_overrides = {}
        for model_key in cfg['experiments']['model_types']:
            if model_key in active_hyperparameters:
                if model_key in ["resnet", "mobilenetv2", "efficientnetb0"]:
                    if f"{model_key}_epochs_override" in cfg.get('test_run', {}):
                        active_hyperparameters[model_key]['epochs'] = cfg['test_run'][f"{model_key}_epochs_override"]
                        print_test_overrides[f"{model_key.capitalize()}-Epochs"] = active_hyperparameters[model_key][
                            'epochs']
                if model_key == "rf":
                    if "rf_n_estimators_override" in cfg.get('test_run', {}):
                        active_hyperparameters['rf']['n_estimators'] = cfg['test_run']['rf_n_estimators_override']
                        print_test_overrides["RF-N_est"] = active_hyperparameters['rf']['n_estimators']
        print(
            f"!!! TEST MODE ENABLED for validation folds: {folds_to_process_numbers}. SampleFrac for HOG CSVs (if RF & test)={sample_fraction_for_test_run:.3f}, Overrides: {print_test_overrides} !!!")
    else:
        print(
            f"--- FULL MODE ENABLED for validation folds: {folds_to_process_numbers}. Using full samples from CSVs and hyperparameters from config (RF HOG input might be sampled via full_mode_sample_fraction). ---")

    temp_sampled_splits_dir = PROJECT_ROOT / cfg['paths']['splits_dir'] / "temp_orchestrator_sampled_splits"
    global_splits_dir = PROJECT_ROOT / cfg['paths']['splits_dir']
    base_feature_dir_path = PROJECT_ROOT / cfg['paths']['base_feature_dir']
    original_rgb_data_dir = PROJECT_ROOT / cfg['paths']['base_data_dir'] / "original"

    ml_models_requiring_hog = [mt for mt in cfg['experiments']['model_types'] if mt in ["rf", "svm"]]
    all_final_hog_input_stems_for_rf = []
    temporary_sampling_directory_needed_for_hog = False

    if ml_models_requiring_hog:
        print(
            f"ML models requiring HOG features detected: {ml_models_requiring_hog}. Preparing HOG features for all {k_folds_total_from_config} original folds if necessary.")

        if "rf" in ml_models_requiring_hog and not is_test_run and cfg.get('hyperparameters', {}).get('rf', {}).get(
                'full_mode_sample_fraction', 1.0) < 1.0:
            temporary_sampling_directory_needed_for_hog = True
        if is_test_run and sample_fraction_for_test_run < 1.0:
            temporary_sampling_directory_needed_for_hog = True

        if temporary_sampling_directory_needed_for_hog:
            if temp_sampled_splits_dir.exists(): shutil.rmtree(temp_sampled_splits_dir)
            temp_sampled_splits_dir.mkdir(parents=True, exist_ok=True)
            print(f"Temporary sampled splits for HOG input will be stored in: {temp_sampled_splits_dir}")

        for fold_idx_orig in range(1, k_folds_total_from_config + 1):
            original_fold_csv_stem = f"fold_{fold_idx_orig}"
            original_split_csv_path = global_splits_dir / f"{original_fold_csv_stem}.csv"

            current_hog_input_csv_stem = original_fold_csv_stem
            current_splits_dir_for_hog_cmd = global_splits_dir
            sampling_fraction_for_this_hog_csv = 1.0
            apply_sampling_for_this_hog_csv = False

            if "rf" in ml_models_requiring_hog:
                if not is_test_run:
                    rf_fm_frac = cfg.get('hyperparameters', {}).get('rf', {}).get('full_mode_sample_fraction', 1.0)
                    if rf_fm_frac < 1.0:
                        sampling_fraction_for_this_hog_csv = rf_fm_frac
                        apply_sampling_for_this_hog_csv = True
                else:
                    if sample_fraction_for_test_run < 1.0:
                        sampling_fraction_for_this_hog_csv = sample_fraction_for_test_run
                        apply_sampling_for_this_hog_csv = True

            if apply_sampling_for_this_hog_csv:
                temp_csv_path = create_temporary_split_csv(
                    original_split_csv_path,
                    temp_sampled_splits_dir,
                    sampling_fraction_for_this_hog_csv,
                    cfg['cv']['seed'],
                    fold_idx_orig,
                    global_splits_dir
                )
                current_hog_input_csv_stem = temp_csv_path.stem
                current_splits_dir_for_hog_cmd = temp_sampled_splits_dir
                print(
                    f"    HOG for original fold {fold_idx_orig} will use sampled CSV: {temp_csv_path.name} (frac: {sampling_fraction_for_this_hog_csv:.3f})")

            all_final_hog_input_stems_for_rf.append(current_hog_input_csv_stem)

            for target_cs_for_hog in cfg['experiments']['colorspaces']:
                hog_command = [
                    python_executable, str(SRC_ROOT / "ml/feature_extractor/hog.py"),
                    "--image_data_root", str(original_rgb_data_dir),
                    "--target_colorspace", target_cs_for_hog,
                    "--splits_dir", str(current_splits_dir_for_hog_cmd),
                    "--fold_id", current_hog_input_csv_stem,
                    "--output_dir_base", str(base_feature_dir_path)
                ]
                expected_hog_file = base_feature_dir_path / f"hog_{target_cs_for_hog}" / f"{current_hog_input_csv_stem}_hog_features.pkl"
                if expected_hog_file.exists():
                    tqdm.write(
                        f"    HOG features for (TargetCS: {target_cs_for_hog}, CSV Stem: {current_hog_input_csv_stem}) exist. Skipping.")
                else:
                    run_command(hog_command, f"HOG ({target_cs_for_hog}, CSV: {current_hog_input_csv_stem[:15]})",
                                timeout_seconds=current_run_timeout, pbar=None)

        print(f"  All HOG input CSV stems available for use by RF: {all_final_hog_input_stems_for_rf}")
    else:
        print("No models requiring HOG features. Skipping HOG preparation.")

    num_colorspaces = len(cfg['experiments']['colorspaces'])
    num_model_types = len(cfg['experiments']['model_types'])
    total_iterations = num_colorspaces * num_model_types * len(folds_to_process_numbers)

    overall_pbar_desc = f"CV (ValFolds: {','.join(map(str, folds_to_process_numbers))})"
    experiment_pbar = tqdm(total=total_iterations, desc=overall_pbar_desc, unit="task", position=0, leave=True)

    for designated_val_fold_num in folds_to_process_numbers:
        for colorspace in cfg['experiments']['colorspaces']:
            for model_type in cfg['experiments']['model_types']:

                current_model_hparams_for_run_cfg = active_hyperparameters.get(model_type, {})
                if not current_model_hparams_for_run_cfg and model_type in cfg['experiments']['model_types']:
                    tqdm.write(
                        f"    WARNING: Hyperparams for model '{model_type}' not found. Skipping for {colorspace}.",
                        file=sys.stderr)
                    experiment_pbar.update(1)
                    continue

                timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                exp_name_suffix_parts = [f"val{designated_val_fold_num}"]
                if not is_test_run:
                    exp_name_suffix_parts.insert(0, "fullrun")
                if is_test_run:
                    exp_name_suffix_parts.insert(0, "testrun")
                    exp_name_suffix_parts.append(f"frac{sample_fraction_for_test_run:.2f}")

                if model_type == "rf" and not is_test_run:
                    rf_hp = cfg.get('hyperparameters', {}).get('rf', {})
                    rf_fm_frac = rf_hp.get('full_mode_sample_fraction')
                    if rf_fm_frac is not None and rf_fm_frac < 1.0:
                        exp_name_suffix_parts.append(f"rfHOGfrac{rf_fm_frac:.2f}")

                exp_name_suffix = "_".join(exp_name_suffix_parts)
                exp_name = f"{colorspace}_{model_type}_exp_{timestamp}_{exp_name_suffix}"
                current_experiment_output_dir = PROJECT_ROOT / cfg['paths']['base_output_dir'] / model_type / exp_name
                current_experiment_output_dir.mkdir(parents=True, exist_ok=True)

                experiment_pbar.set_description(
                    f"{overall_pbar_desc} (CurVal:{designated_val_fold_num}, CS:{colorspace}, Model:{model_type})")
                run_specific_output_subdir_name = f"fold_{designated_val_fold_num}"

                training_cmd_args = []
                known_cnn_models = ["resnet", "mobilenetv2", "efficientnetb0"]

                if model_type in known_cnn_models:
                    run_specific_cnn_cfg = {
                        'hyperparameters': {model_type: current_model_hparams_for_run_cfg},
                        'paths': {
                            'data_dir': str(original_rgb_data_dir),
                            'splits_dir': str(global_splits_dir),
                            'output_root': str(current_experiment_output_dir)
                        },
                        'cv': {
                            'k_folds_total': k_folds_total_from_config,
                            'current_fold_num_to_run': designated_val_fold_num,
                            'seed': cfg['cv']['seed']
                        },
                        'target_colorspace': colorspace,
                    }
                    if is_test_run and sample_fraction_for_test_run < 1.0:
                        run_specific_cnn_cfg['sample_fraction_for_test'] = sample_fraction_for_test_run

                    temp_run_config_path = current_experiment_output_dir / f"run_cfg_{model_type}_{run_specific_output_subdir_name}.yaml"
                    with open(temp_run_config_path, 'w', encoding='utf-8') as f_cfg:
                        yaml.dump(run_specific_cnn_cfg, f_cfg)

                    train_script_path = SRC_ROOT / "dl" / model_type / "train.py"
                    if not train_script_path.exists():
                        tqdm.write(f"    ERROR: Train script {train_script_path} for {model_type} not found. Skipping.",
                                   file=sys.stderr)
                        experiment_pbar.update(1)
                        continue
                    training_cmd_args = [python_executable, str(train_script_path), "--config",
                                         str(temp_run_config_path)]

                elif model_type in ml_models_requiring_hog:
                    if not all_final_hog_input_stems_for_rf:
                        tqdm.write(f"    Skipping {model_type} as HOG input stems list is empty.", file=sys.stderr)
                        experiment_pbar.update(1)
                        continue

                    val_hog_stem_for_rf = None
                    expected_original_stem_part = f"fold_{designated_val_fold_num}"

                    if 0 <= (designated_val_fold_num - 1) < len(all_final_hog_input_stems_for_rf):
                        val_hog_stem_for_rf = all_final_hog_input_stems_for_rf[designated_val_fold_num - 1]
                        if not val_hog_stem_for_rf.startswith(expected_original_stem_part):
                            print(
                                f"    Warning: Mismatch in expected HOG stem for fold {designated_val_fold_num}. Found '{val_hog_stem_for_rf}' at index, expected prefix '{expected_original_stem_part}'. Using it anyway.")

                    if val_hog_stem_for_rf is None:
                        tqdm.write(
                            f"    ERROR: Could not determine validation HOG stem for RF for val_fold_num {designated_val_fold_num} from available stems {all_final_hog_input_stems_for_rf}. Skipping {model_type}.",
                            file=sys.stderr)
                        experiment_pbar.update(1)
                        continue

                    fold_specific_output_dir_ml = current_experiment_output_dir / run_specific_output_subdir_name

                    rf_hparams = active_hyperparameters.get("rf", {})
                    common_ml_args = [
                        "--output_dir", str(fold_specific_output_dir_ml),
                        "--base_feature_dir", str(base_feature_dir_path),
                        "--colorspace", colorspace,
                        "--val_fold_hog_stem", val_hog_stem_for_rf,
                        "--all_hog_stems_json", json.dumps(all_final_hog_input_stems_for_rf),
                        "--random_state", str(rf_hparams.get('random_state', 42)),
                        "--n_estimators", str(rf_hparams.get('n_estimators', 100)),
                        "--min_samples_split", str(rf_hparams.get('min_samples_split', 2)),
                        "--min_samples_leaf", str(rf_hparams.get('min_samples_leaf', 1))
                    ]
                    if rf_hparams.get('max_depth') is not None:
                        common_ml_args.extend(["--max_depth", str(rf_hparams.get('max_depth'))])

                    if model_type == "rf":
                        training_cmd_args = [python_executable, str(SRC_ROOT / "ml/rf/train.py")] + common_ml_args
                    else:
                        tqdm.write(f"    Logic for ML model {model_type} not fully implemented. Skipping.",
                                   file=sys.stderr)
                        experiment_pbar.update(1)
                        continue
                else:
                    tqdm.write(f"    Skipping model type '{model_type}' - no specific training logic.", file=sys.stderr)
                    experiment_pbar.update(1)
                    continue

                if training_cmd_args:
                    run_command(training_cmd_args,
                                f"Training ({model_type}, CS:{colorspace}, ValF:{designated_val_fold_num})",
                                timeout_seconds=current_run_timeout,
                                pbar=experiment_pbar)
                else:
                    tqdm.write(
                        f"    No training command generated for {model_type}, {colorspace}, ValFold: {designated_val_fold_num}. Skipping.",
                        file=sys.stderr)

                experiment_pbar.update(1)

    experiment_pbar.set_description(f"{overall_pbar_desc} (COMPLETED)")
    experiment_pbar.set_postfix_str("All tasks complete!")
    experiment_pbar.close()

    if temporary_sampling_directory_needed_for_hog and temp_sampled_splits_dir.exists():
        print(f"\nCleaning up temporary sampled split directory: {temp_sampled_splits_dir}")
        try:
            shutil.rmtree(temp_sampled_splits_dir)
        except OSError as e:
            print(f"Warning: Could not remove temporary directory {temp_sampled_splits_dir}: {e}")

    print("\n--- All Experiments Orchestration Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments for Image Analysis.")
    parser.add_argument("--mode", type=str, default="full", choices=["test", "full"],
                        help="Run mode: 'test' for quick debug runs, 'full' for complete training.")
    parser.add_argument("--run_folds", type=int, nargs='*', default=None,
                        help="Specify validation fold numbers to run (e.g., --run_folds 1 2 5). Runs all folds if not specified.")

    cli_args = parser.parse_args()
    main(run_mode=cli_args.mode, requested_folds=cli_args.run_folds)