import os
import logging
import torch
import sys
from transformers import AutoTokenizer
from IRutils.load_data import calculate_percentiles
from IRutils.inference_plus import evaluate, evaluate_average_ensemble, evaluate_conditional_ensemble, evaluate_weighted_average_ensemble, write_results
from IRutils.load_data import load, preprocess
from IRutils.models import load_model, load_models
from IRutils.plotting_utils import *
from IRutils.weight_optimizer import precompute_validation_scores, find_optimal_weights_config
from ir_measures import nDCG, RR, R, P
from IRutils.dataset import encode_corpus


logging.disable(logging.WARNING) # General disable for others if needed

def run(model_name, dataset_name, metrics, device, length_setting='full', max_len_doc=512, random_state=42):
    results = {'baseline': {}, 'ens-avg': {}, 'ens-select': {}, 'ens-weighted': {},
               'ens-learned-weighted': {}}  # Added new key

    model_dir = f'models\\{model_name}\\{dataset_name}'
    baseline_path = os.path.join(model_dir, 'full_queries.pth')
    results_save_dir = os.path.join('results', model_name.replace('/', os.sep), dataset_name)

    print(f'Loading baseline model from {baseline_path}...')
    baseline_model = load_model(baseline_path, model_name, device) # load baseline model
    models = load_models(model_dir, model_name, device)

    train_available, docs, queries, qrels, docs_test, queries_test, qrels_test = load(dataset_name)
    print('Loading complete!')

    query_lengths = [len(txt.split()) for txt in list(queries.values())]
    t1, t2 = calculate_percentiles(query_lengths)  # get query length thresholds

    if train_available:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
            queries, docs, qrels, model_name, length_setting, train_available,
            queries_test=queries_test, qrels_test=qrels_test,
            max_len_doc=max_len_doc, random_state=random_state, for_eval=True)
    else:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
            queries, docs, qrels, model_name, length_setting, train_available,
            max_len_doc=max_len_doc, random_state=random_state, for_eval=True)

    print('Preprocessing complete!')

    all_models = {}

    for name, model in models.items():
        all_models[name] = model

    all_models['full'] = baseline_model

    # ---------------------------------------------------------------
    # Encode corpus embeddings once

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_embeddings = {}

    save_dir = os.path.join('corpus_embeddings', model_name, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for model in all_models:
        print(f'Creating corpus embeddings from {model} model...')
        save_path = os.path.join(save_dir, f'{model}.pt')
        if os.path.exists(save_path):
            try:
                # map_location ensures tensors are loaded onto the desired device
                embeddings = torch.load(save_path, map_location=device)
                print(f"Loaded {len(embeddings)} embeddings successfully.")
                if embeddings and device:
                     # Verify one tensor's device
                     first_key = next(iter(embeddings))
                     print(f"Embeddings loaded onto device: {embeddings[first_key].device}")
                all_embeddings[model] = embeddings
            except Exception as e:
                print(f"Error loading embeddings from {save_path}: {e}")
                raise
        else:
            corpus_embeddings = encode_corpus(docs, all_models[model], tokenizer, device=device)
            all_embeddings[model] = corpus_embeddings
            try:
                torch.save(corpus_embeddings, save_path)
                print(f"Embeddings saved successfully.")
            except Exception as e:
                print(f"Error saving embeddings to {save_path}: {e}")
                raise

    # ---------------------------------------------------------------
    # Perform ranking and evaluation on baseline
    doc_embeddings = all_embeddings['full']

    if train_available:
        baseline_metrics, baseline_per_query, baseline_run = evaluate(baseline_model, tokenizer, queries_test,
                                                                      doc_embeddings, qrels_test, device,max_length=max_len_doc)
    else:
        baseline_metrics, baseline_per_query, baseline_run = evaluate(baseline_model, tokenizer, split_queries_test, doc_embeddings, split_qrels_test, device, max_length=max_len_doc)

    for metric in metrics:
        print(f'Metric {metric} score: {baseline_metrics[metric]:.4f}')

    results['baseline'] = baseline_metrics

    # Save baseline results
    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path_baseline = os.path.join(save_dir, 'baseline.txt')
    write_results(baseline_metrics, save_path_baseline, model_name, dataset_name, length_setting)
    print(f"Baseline results saved to {save_path_baseline}")
    del baseline_model # Free memory
    torch.cuda.empty_cache() # Clear cache if using GPU
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using average ensemble method

    models = load_models(model_dir, model_name, device)

    if train_available:
        avg_metrics, avg_per_query, avg_run = evaluate_average_ensemble(models, tokenizer, queries_test, all_embeddings, qrels_test, device, max_len_doc)
    else:
       avg_metrics, avg_per_query, avg_run = evaluate_average_ensemble(models, tokenizer, split_queries_test, all_embeddings, split_qrels_test, device, max_len_doc)

    for metric in metrics:
        print(f'Metric {metric} score: {avg_metrics[metric]:.4f}')

    results['ens-avg'] = avg_metrics

    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ensemble-avg.txt')

    write_results(avg_metrics, save_path, model_name, dataset_name, length_setting)
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using selective ensemble method

    if train_available:
        cond_metrics, cond_per_query, cond_run = evaluate_conditional_ensemble(models, tokenizer, queries_test, all_embeddings, qrels_test, t1, t2, device, max_len_doc)
    else:
        cond_metrics, cond_per_query, cond_run = evaluate_conditional_ensemble(models, tokenizer, split_queries_test, all_embeddings, split_qrels_test, t1, t2, device, max_len_doc)

    for metric in metrics:
        print(f'Metric {metric} score: {cond_metrics[metric]:.4f}')

    results['ens-select'] = cond_metrics

    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ensemble-selective.txt')

    write_results(cond_metrics, save_path, model_name, dataset_name, length_setting)
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using weighted ensemble method

    model_dir = f'models\\{model_name}\\{dataset_name}'
    models = load_models(model_dir, model_name, device)

    weights_config = {'short': [0.5, 0.25, 0.25], # Weights for [short, medium, long] models when query is short
                      'medium': [0.25, 0.5, 0.25],# Weights when query is medium
                      'long': [0.25, 0.25, 0.5]   # Weights when query is long
                     }

    if train_available:
        weighted_metrics, weighted_per_query, weighted_run = evaluate_weighted_average_ensemble(models, tokenizer, queries_test, all_embeddings, weights_config, qrels_test, t1, t2, device, max_len_doc)
    else:
        weighted_metrics, weighted_per_query, weighted_run = evaluate_weighted_average_ensemble(models, tokenizer, split_queries_test, all_embeddings, weights_config, split_qrels_test, t1, t2, device, max_len_doc)

    for metric in metrics:
        print(f'Metric {metric} score: {weighted_metrics[metric]:.4f}')

    results['ens-weighted'] = weighted_metrics

    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ensemble-weighted.txt')

    write_results(weighted_metrics, save_path, model_name, dataset_name, length_setting)
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Peform ranking and evaluation using weighted ensemble + regression method
    # Precompute scores on the validation set
    print("\n--- Optimizing Ensemble Weights using Validation Set ---")
    models_all = load_models(model_dir, model_name, device) # Reload or ensure models are available
    precomputed_val_scores = precompute_validation_scores(models_all, val_loader, device)

    # Find the single optimal weights configuration using the validation set
    weight_opt_trials = 1000  # number of config trials
    metric_to_optimize_weights = nDCG @ 10

    learned_weights_config = find_optimal_weights_config(
        precomputed_val_scores,
        query_val,
        qrels_val,
        t1, t2,
        metric_to_optimize=metric_to_optimize_weights, # NDCG@10
        n_trials=weight_opt_trials,
        random_state=random_state
    )

    print("\nLearned Weights Config:")
    print(learned_weights_config)

    # Evaluate on the TEST set using the single learned weights config
    print("\n--- Evaluating on TEST set using LEARNED weights configuration ---")
    # Models should still be loaded in models_all
    if train_available:
        learned_weighted_metrics, learned_weighted_per_query, learned_weighted_run = evaluate_weighted_average_ensemble(models, tokenizer, queries_test, all_embeddings, learned_weights_config, qrels_test, t1, t2, device, max_len_doc)
    else:
        learned_weighted_metrics, learned_weighted_per_query, learned_weighted_run = evaluate_weighted_average_ensemble(models, tokenizer, split_queries_test, all_embeddings, learned_weights_config, split_qrels_test, t1, t2, device, max_len_doc)

    print("\nFinal Test Set Performance with Learned Weights:")
    for metric in metrics:
         print(f'Metric {metric} score: {learned_weighted_metrics[metric]:.4f}')

    results['ens-learned-weighted'] = learned_weighted_metrics

    # Save learned weighted results
    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ensemble-learned_weighted.txt')
    write_results(learned_weighted_metrics, save_path, model_name, dataset_name, length_setting) # Updated description
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Plot results

    create_comparison_plot(results, metrics, model_name, dataset_name, save_dir)
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform statistical t-test on results

    from IRutils.inference_plus import compare_models_with_ttest, write_ttest_results

    # --- 9. Run t-tests and save results ---

    # Perform t-tests between baseline and all methods
    model_names = ["Average Ensemble", "Conditional Ensemble", "Weighted Ensemble", "Regression Weighted Ensemble"]
    model_results = [
        (avg_metrics, avg_per_query, avg_run),
        (cond_metrics, cond_per_query, cond_run),
        (weighted_metrics, weighted_per_query, weighted_run),
        (learned_weighted_metrics, learned_weighted_per_query, learned_weighted_run)
    ]

    ttest_df = compare_models_with_ttest(
        (baseline_metrics, baseline_per_query, baseline_run),
        model_results,
        metrics,
        model_names
    )

    ttest_save_path = os.path.join(save_dir, 'ttest_results.txt')
    # Save t-test results
    write_ttest_results(ttest_df, ttest_save_path, "Baseline")

    # You can also print the DataFrame for a quick view
    print("\nT-Test Results Summary:")
    print(ttest_df[['Model', 'Metric', 'P-Value', 'Significant', 'Improvement %']])
    # ---------------------------------------------------------------


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the models and datasets to run
    run_models = ['distilroberta-base']
    run_datasets = ['fiqa', 'quora']
    # run_datasets = ['fiqa', 'quora']
    metrics = [
        nDCG@100,  # Quality of top 10 results (standard)
        R@100,  # Recall within top 100
        RR
    ]

    # Example for multiple runs (uncomment above lines and comment single run lines)
    print("Starting ensemble evaluation runs...")
    total_runs = len(run_models) * len(run_datasets)
    current_run = 0

    for model in run_models:
        for dataset in run_datasets:
            current_run += 1
            print(f"\n>>> Starting Run {current_run}/{total_runs} <<<")
            try:
                if model == 'microsoft/MiniLM-L12-H384-uncased' and dataset == 'fiqa':
                    continue
                run(model, dataset, metrics, device)
            except Exception as e:
                print(f"!!! CRITICAL ERROR during run for {model} on {dataset}: {e} !!!")
                print("!!! Skipping to next run !!!")
            # Optional: Add delay or clear memory if needed between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll ensemble evaluation runs completed.")