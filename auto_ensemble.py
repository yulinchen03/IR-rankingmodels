import os
import sys
import torch
import logging
import optuna  # Import optuna for logging control

# Assuming IRutils is importable (e.g., in the same directory or installed)
from IRutils.load_data import calculate_percentiles, load, preprocess
from IRutils.inference import evaluate, evaluate_average_ensemble, evaluate_conditional_ensemble, evaluate_weighted_average_ensemble, write_results
from IRutils.models import load_model, load_models
from IRutils.plotting_utils import create_comparison_plot
from IRutils.weight_optimizer import precompute_validation_scores, find_optimal_weights_config
from ir_measures import nDCG, P, R, RR

METRICS = [nDCG @ 3, nDCG @ 5, nDCG @ 10, RR, P @ 1, R @ 1, R @ 3, R @ 5, R @ 10]
MAX_LEN_DOC = 512
RANDOM_STATE = 42
METRIC_TO_OPTIMIZE_WEIGHTS = nDCG @ 10 # Choose the metric to optimize weights for
WEIGHT_OPT_TRIALS = 1000 # Number of Optuna trials per category (adjust as needed)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.disable(logging.WARNING) # General disable for others if needed

def run(model_name, dataset_name):
    """
    Performs baseline and ensemble evaluations for a given model and dataset.

    Args:
        model_name (str): The name of the base model (e.g., 'huawei-noah/TinyBERT_General_4L_312D').
        dataset_name (str): The name of the dataset (e.g., 'fiqa').
    """
    print(f"\n--- Running Ensemble Evaluation ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("-" * 30)

    results = {'baseline': {}, 'ens-avg': {}, 'ens-select': {}, 'ens-weighted': {}, 'ens-learned-weighted': {}}
    length_setting_baseline = 'full' # Baseline always uses the 'full' model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_dir = os.path.join('models', model_name.replace('/', os.sep), dataset_name) # Use os.sep for path compatibility
    save_dir = os.path.join('results', model_name.replace('/', os.sep), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Load Data & Preprocess ---
    print("\n--- Loading and Preprocessing Data ---")
    try:
        train_available, docs, queries, qrels, docs_test, queries_test, qrels_test = load(dataset_name)
        print('Dataset loading complete!')

        query_lengths = [len(txt.split()) for txt in list(queries.values())]
        t1, t2 = calculate_percentiles(query_lengths)
        print(f"Query length percentiles: t1={t1}, t2={t2}")

        # Preprocess using 'full' setting initially to get all loaders needed
        # Note: The preprocess function internally handles splitting based on its length_setting arg,
        # but we need the validation set split based on the *original* full query set for weight optimization.
        # We also need the test set split based on the *original* full test set.
        print("Preprocessing data (this might take a moment)...")
        if train_available:
            train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
                queries, docs, qrels, model_name, 'full', train_available,
                queries_test=queries_test, docs_test=docs_test, qrels_test=qrels_test,
                max_len_doc=MAX_LEN_DOC, random_state=RANDOM_STATE
            )
        else:
             # If no train split, validation comes from the main set, test is the rest
             # Note: Original notebook logic reused test_loader/split_qrels_test for both baseline and ensembles.
             # Ensure preprocess handles this split correctly. The current IRutils seems designed for train/val/test or train_test_split.
             # We will assume preprocess correctly gives us a val_loader and test_loader derived from the input queries/qrels.
             # Let's call preprocess once to get all splits needed.
             # We pass 'full' to ensure it processes all queries initially.
            train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
                queries, docs, qrels, model_name, 'full', train_available,
                max_len_doc=MAX_LEN_DOC, random_state=RANDOM_STATE
            )
            # If no train_available, the 'test' data used later corresponds to split_queries_test/split_qrels_test
            queries_test = split_queries_test # Use these for ensemble methods if no predefined test set
            qrels_test = split_qrels_test


        print('Preprocessing complete!')

    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}")
        return # Stop processing this combination

    # Determine which qrels/queries to use for final test evaluation based on availability
    final_test_qrels = qrels_test if train_available else split_qrels_test
    final_test_queries = queries_test if train_available else split_queries_test


    # --- 2. Load All Ensemble Models ---
    # Load models needed for ensemble methods first
    print("\n--- Loading Ensemble Models ---")
    try:
        # Ensure model directory structure matches expectations (e.g., models/model_name/dataset_name/short_queries.pth)
        ensemble_models = load_models(model_dir, model_name, device)
        if not ensemble_models:
             print(f"Warning: No ensemble models found in {model_dir}. Skipping ensemble evaluations.")
             # We could potentially still run baseline, but let's skip all if ensembles are expected.
             return
        print(f"Successfully loaded {len(ensemble_models)} ensemble models.")
    except Exception as e:
        print(f"Error loading ensemble models from {model_dir}: {e}")
        return # Stop if models can't be loaded


    # --- 3. Evaluate Baseline Model ---
    print("\n--- Evaluating Baseline Model (full_queries) ---")
    try:
        model_path_baseline = os.path.join(model_dir, f'{length_setting_baseline}_queries.pth')
        if not os.path.exists(model_path_baseline):
             print(f"Baseline model not found at {model_path_baseline}. Skipping baseline evaluation.")
        else:
            model_baseline = load_model(model_path_baseline, model_name, device)
            metric_scores_baseline = evaluate(model_baseline, test_loader, device, final_test_qrels)

            print("Baseline Scores:")
            for metric in METRICS:
                print(f'  Metric {metric} score: {metric_scores_baseline.get(metric, float("nan")):.4f}')
            results['baseline'] = metric_scores_baseline

            # Save baseline results
            save_path_baseline = os.path.join(save_dir, 'baseline.txt')
            write_results(metric_scores_baseline, save_path_baseline, model_name, dataset_name, length_setting_baseline)
            print(f"Baseline results saved to {save_path_baseline}")
            del model_baseline # Free memory
            torch.cuda.empty_cache() # Clear cache if using GPU
    except Exception as e:
        print(f"Error during baseline evaluation: {e}")


    # --- 4. Evaluate Ensemble - Average ---
    print("\n--- Evaluating Ensemble (Average) ---")
    try:
        metric_scores_avg = evaluate_average_ensemble(ensemble_models, test_loader, device, final_test_qrels)
        print("Ensemble Average Scores:")
        for metric in METRICS:
            print(f'  Metric {metric} score: {metric_scores_avg.get(metric, float("nan")):.4f}')
        results['ens-avg'] = metric_scores_avg

        save_path_avg = os.path.join(save_dir, 'ensemble-avg.txt')
        write_results(metric_scores_avg, save_path_avg, model_name, dataset_name, "ensemble-average")
        print(f"Ensemble average results saved to {save_path_avg}")
    except Exception as e:
        print(f"Error during average ensemble evaluation: {e}")

    # --- 5. Evaluate Ensemble - Selective ---
    print("\n--- Evaluating Ensemble (Selective) ---")
    try:
        metric_scores_select = evaluate_conditional_ensemble(ensemble_models, t1, t2, test_loader, device, final_test_qrels, final_test_queries)
        print("Ensemble Selective Scores:")
        for metric in METRICS:
            print(f'  Metric {metric} score: {metric_scores_select.get(metric, float("nan")):.4f}')
        results['ens-select'] = metric_scores_select

        save_path_select = os.path.join(save_dir, 'ensemble-selective.txt')
        write_results(metric_scores_select, save_path_select, model_name, dataset_name, "ensemble-selective")
        print(f"Ensemble selective results saved to {save_path_select}")
    except Exception as e:
        print(f"Error during selective ensemble evaluation: {e}")


    # --- 6. Evaluate Ensemble - Weighted (Fixed) ---
    print("\n--- Evaluating Ensemble (Weighted - Fixed) ---")
    try:
        # Using the fixed weights from the notebook
        weights_config_fixed = {
            'short': [0.6, 0.2, 0.2], # Weights for [short, medium, long] models when query is short
            'medium': [0.2, 0.6, 0.2],# Weights when query is medium
            'long': [0.2, 0.2, 0.6]   # Weights when query is long
        }
        print(f"Using fixed weights: {weights_config_fixed}")
        metric_scores_weighted = evaluate_weighted_average_ensemble(ensemble_models, weights_config_fixed, t1, t2, test_loader, device, final_test_qrels, final_test_queries)
        print("Ensemble Weighted (Fixed) Scores:")
        for metric in METRICS:
            print(f'  Metric {metric} score: {metric_scores_weighted.get(metric, float("nan")):.4f}')
        results['ens-weighted'] = metric_scores_weighted

        save_path_weighted = os.path.join(save_dir, 'ensemble-weighted.txt')
        write_results(metric_scores_weighted, save_path_weighted, model_name, dataset_name, "ensemble-weighted-fixed")
        print(f"Ensemble weighted (fixed) results saved to {save_path_weighted}")
    except Exception as e:
        print(f"Error during fixed weighted ensemble evaluation: {e}")


    # --- 7. Evaluate Ensemble - Weighted (Learned) ---
    # Requires validation set - check if val_loader is available
    if val_loader and query_val and qrels_val:
        print("\n--- Optimizing and Evaluating Ensemble (Weighted - Learned) ---")
        try:
            # Precompute scores on the validation set
            print("Precomputing validation scores...")
            # Note: Ensure ensemble_models contains the models in the expected order [short, medium, long]
            # The load_models function should guarantee this if filenames are consistent.
            precomputed_val_scores = precompute_validation_scores(ensemble_models, val_loader, device)
            print("Validation score precomputation complete.")

            # Find the single optimal weights configuration using the validation set
            print(f"Finding optimal weights using Optuna ({WEIGHT_OPT_TRIALS} trials, optimizing for {METRIC_TO_OPTIMIZE_WEIGHTS})...")
            learned_weights_config = find_optimal_weights_config(
                precomputed_val_scores,
                query_val,
                qrels_val,
                t1, t2,
                metric_to_optimize=METRIC_TO_OPTIMIZE_WEIGHTS,
                n_trials=WEIGHT_OPT_TRIALS,
                random_state=RANDOM_STATE
            )
            print("\nLearned Weights Config:")
            print(learned_weights_config)

            # Evaluate on the TEST set using the single learned weights config
            print("\nEvaluating on TEST set using LEARNED weights configuration...")
            metric_scores_learned_w = evaluate_weighted_average_ensemble(ensemble_models, learned_weights_config, t1, t2, test_loader, device, final_test_qrels, final_test_queries)

            print("\nFinal Test Set Performance with Learned Weights:")
            for metric in METRICS:
                 print(f'  Metric {metric} score: {metric_scores_learned_w.get(metric, float("nan")):.4f}')
            results['ens-learned-weighted'] = metric_scores_learned_w

            # Save learned weighted results
            save_path_learned = os.path.join(save_dir, 'ensemble-weighted-reg.txt')
            write_results(metric_scores_learned_w, save_path_learned, model_name, dataset_name, "learned-weighted-config")
            print(f"Ensemble learned weighted results saved to {save_path_learned}")

        except Exception as e:
            print(f"Error during learned weighted ensemble evaluation: {e}")
    else:
        print("\n--- Skipping Learned Weighted Ensemble (Validation data not available) ---")


    # --- 8. Plot Results ---
    print("\n--- Plotting Results ---")
    try:
        # Check if results dictionary has data before plotting
        if any(results.values()):
             plot_save_path = create_comparison_plot(results, METRICS, model_name, dataset_name, save_dir)
             print(f"Comparison plot saved to {plot_save_path}")
        else:
             print("Skipping plot generation as no results were successfully collected.")
    except Exception as e:
        print(f"Error during plotting: {e}")

    print(f"\n--- Finished Ensemble Evaluation for {model_name} on {dataset_name} ---")


if __name__ == "__main__":
    # Define the models and datasets to run
    run_models = ['huawei-noah/TinyBERT_General_4L_312D', 'microsoft/MiniLM-L12-H384-uncased', 'distilbert-base-uncased', 'distilroberta-base']
    run_datasets = ['quora', 'fiqa']

    # Example for multiple runs (uncomment above lines and comment single run lines)
    print("Starting ensemble evaluation runs...")
    total_runs = len(run_models) * len(run_datasets)
    current_run = 0

    for model in run_models:
        for dataset in run_datasets:
            current_run += 1
            print(f"\n>>> Starting Run {current_run}/{total_runs} <<<")
            try:
                run(model, dataset)
            except Exception as e:
                print(f"!!! CRITICAL ERROR during run for {model} on {dataset}: {e} !!!")
                print("!!! Skipping to next run !!!")
            # Optional: Add delay or clear memory if needed between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll ensemble evaluation runs completed.")