import os
import logging
import torch
import sys
import gc # Import garbage collector interface
from transformers import AutoTokenizer, AutoModel # Added AutoModel for explicit loading if needed
from IRutils.load_data import calculate_percentiles
# Ensure these functions are updated as shown below
from IRutils.inference import (
    evaluate,
    evaluate_average_ensemble,
    evaluate_conditional_ensemble,
    evaluate_weighted_average_ensemble,
    write_results,
    compare_models_with_ttest,
    write_ttest_results
)
from IRutils.load_data import load, preprocess
from IRutils.models import load_model, load_models # Assuming these are still needed for ensemble parts
from IRutils.plot_utils import create_comparison_plot
from IRutils.weight_optimizer import precompute_validation_scores, find_optimal_weights_config
from ir_measures import nDCG, RR, R, P
from IRutils.dataset import encode_corpus # Make sure this function exists and works as expected


logging.disable(logging.WARNING) # General disable for others if needed

def run(model_name, dataset_name, metrics, device, length_setting='full', max_len_doc=512, random_state=42):

    results = {'baseline': {}, 'ens-avg': {}, 'ens-select': {}, 'ens-weighted': {},
               'ens-learned-weighted': {}}

    model_dir = f'models/{model_name.replace("/", os.sep)}/{dataset_name}' # Use forward slashes for consistency, os.sep for joining
    baseline_path = os.path.join(model_dir, 'full_queries.pth')
    results_save_dir = os.path.join('results', model_name.replace('/', os.sep), dataset_name)
    embedding_save_dir = os.path.join('corpus_embeddings', model_name.replace('/', os.sep), dataset_name) # Define central embedding dir

    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(embedding_save_dir, exist_ok=True) # Ensure embedding dir exists

    print(f'Loading baseline model from {baseline_path}...')
    # Keep baseline model loaded initially for embedding generation if needed, and evaluation
    baseline_model = load_model(baseline_path, model_name, device)
    # Load ensemble component models - needed for generation and evaluation
    models = load_models(model_dir, model_name, device) # keys: short, medium, long

    train_available, docs, queries, qrels, docs_test, queries_test, qrels_test = load(dataset_name)
    print('Loading complete!')

    query_lengths = [len(txt.split()) for txt in list(queries.values())]
    t1, t2 = calculate_percentiles(query_lengths)

    # We need the tokenizer for embedding generation and potentially evaluation functions
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the data
    if train_available:
        # Assuming preprocess handles splitting test set if needed, and returns val set
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
            queries, docs, qrels, model_name, length_setting, train_available,
            queries_test=queries_test, qrels_test=qrels_test,
            max_len_doc=max_len_doc, random_state=random_state, for_eval=True)
        eval_queries = queries_test # Use the original full test set queries if train available
        eval_qrels = qrels_test
    else:
        # If no train set, preprocess splits original data for test/val
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, query_val, qrels_val = preprocess(
            queries, docs, qrels, model_name, length_setting, train_available,
            max_len_doc=max_len_doc, random_state=random_state, for_eval=True)
        eval_queries = split_queries_test # Use the split test set queries if no train available
        eval_qrels = split_qrels_test

    print('Preprocessing complete!')

    # Combine models for easier iteration during embedding generation check
    all_models_for_check = {'full': baseline_model, **models}

    # ---------------------------------------------------------------
    # Ensure Corpus Embeddings Exist (Generate if missing)
    print("\n--- Ensuring Corpus Embeddings Exist ---")
    for model_key, model_obj in all_models_for_check.items():
        save_path = os.path.join(embedding_save_dir, f'{model_key}.pt')
        if not os.path.exists(save_path):
            print(f'Generating corpus embeddings for {model_key} model...')
            try:
                # Ensure the model object is valid and on the correct device
                if model_obj is None:
                     raise ValueError(f"Model object for key '{model_key}' is None.")
                model_obj.to(device) # Make sure model is on the right device
                model_obj.eval()    # Set to evaluation mode

                corpus_embeddings = encode_corpus(docs, model_obj, tokenizer, device=device, max_length=max_len_doc) # Pass max_len_doc if needed by encode_corpus
                torch.save(corpus_embeddings, save_path)
                print(f"Embeddings for {model_key} saved successfully to {save_path}")
                # --- Crucial: Free memory ---
                del corpus_embeddings
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"!!! ERROR generating embeddings for {model_key} to {save_path}: {e} !!!")
                # Decide how to handle: raise error, skip, etc.
                raise # Re-raise to stop execution if embedding generation fails
        else:
            print(f"Embeddings for {model_key} already exist at {save_path}")

    # loading embeddings inside each evaluate function.
    # ---------------------------------------------------------------
    # Perform ranking and evaluation on baseline
    print("\n--- Evaluating Baseline Model ---")
    baseline_embedding_path = os.path.join(embedding_save_dir, 'full.pt')
    baseline_metrics, baseline_per_query, baseline_run = evaluate(
        model=baseline_model,
        tokenizer=tokenizer,
        queries=eval_queries,
        doc_embeddings_path=baseline_embedding_path, # Pass PATH
        qrels=eval_qrels,
        device=device,
        max_length=max_len_doc
    )

    # --- Free baseline model memory ---
    print("Clearing baseline model from memory...")
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using average ensemble method
    print("\n--- Evaluating Average Ensemble ---")
    # Models dict ('short', 'medium', 'long') should still be loaded
    avg_metrics, avg_per_query, avg_run = evaluate_average_ensemble(
        models=models,
        tokenizer=tokenizer,
        queries=eval_queries,
        embedding_save_dir=embedding_save_dir, # Pass DIR
        qrels=eval_qrels,
        device=device,
        max_len_doc=max_len_doc
    )

    # Memory for embeddings used here is cleaned up inside evaluate_average_ensemble
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using selective ensemble method
    print("\n--- Evaluating Conditional Ensemble ---")
    cond_metrics, cond_per_query, cond_run = evaluate_conditional_ensemble(
        models=models,
        tokenizer=tokenizer,
        queries=eval_queries,
        embedding_save_dir=embedding_save_dir, # Pass DIR
        qrels=eval_qrels,
        t1=t1, t2=t2,
        device=device,
        max_len_doc=max_len_doc
    )

    # Memory cleanup inside evaluate_conditional_ensemble
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using weighted ensemble method
    print("\n--- Evaluating Weighted Ensemble ---")
    # Define fixed weights
    weights_config = {'short': [0.5, 0.25, 0.25], 'medium': [0.25, 0.5, 0.25], 'long': [0.25, 0.25, 0.5]}

    weighted_metrics, weighted_per_query, weighted_run = evaluate_weighted_average_ensemble(
        models=models,
        tokenizer=tokenizer,
        queries=eval_queries,
        embedding_save_dir=embedding_save_dir, # Pass DIR
        weights_config=weights_config,
        qrels=eval_qrels,
        t1=t1, t2=t2,
        device=device,
        max_len_doc=max_len_doc
    )

    # Memory cleanup inside evaluate_weighted_average_ensemble
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform ranking and evaluation using learned weighted ensemble method
    print("\n--- Optimizing Ensemble Weights using Validation Set ---")
    # Assuming models dict ('short', 'medium', 'long') is still loaded.
    precomputed_val_scores = precompute_validation_scores(models, val_loader, device)

    weight_opt_trials = 1000  # number of configurations to sweep
    metric_to_optimize_weights = nDCG @ 100 # Example metric

    learned_weights_config = find_optimal_weights_config(
        precomputed_val_scores,
        query_val, qrels_val, # Use validation queries/qrels
        t1, t2,
        metric_to_optimize=metric_to_optimize_weights,
        n_trials=weight_opt_trials,
        random_state=random_state
    )

    print("\nLearned Weights Config:")
    print(learned_weights_config)
    del precomputed_val_scores # Free memory from validation scores
    gc.collect()

    print("\n--- Evaluating on TEST set using LEARNED weights ---")
    learned_weighted_metrics, learned_weighted_per_query, learned_weighted_run = evaluate_weighted_average_ensemble(
        models=models,
        tokenizer=tokenizer,
        queries=eval_queries,
        embedding_save_dir=embedding_save_dir,
        weights_config=learned_weights_config, # Use LEARNED weights
        qrels=eval_qrels,
        t1=t1, t2=t2,
        device=device,
        max_len_doc=max_len_doc
    )

    # Memory cleanup inside evaluate_weighted_average_ensemble
    # ---------------------------------------------------------------

    # --- Free ensemble models memory ---
    print("Clearing ensemble models from memory...")
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    # Plot results
    print("\n--- Plotting Results ---")
    create_comparison_plot(results, metrics, model_name, dataset_name, results_save_dir)
    print(f"Comparison plot saved in {results_save_dir}")
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Perform statistical t-test on results
    print("\n--- Performing T-Tests ---")
    model_names_ttest = ["Average Ensemble", "Conditional Ensemble", "Weighted Ensemble", "Regression Weighted Ensemble"]
    model_results_ttest = [
        (avg_metrics, avg_per_query, avg_run),
        (cond_metrics, cond_per_query, cond_run),
        (weighted_metrics, weighted_per_query, weighted_run),
        (learned_weighted_metrics, learned_weighted_per_query, learned_weighted_run)
    ]

    # Ensure baseline results are not empty before comparing
    if baseline_metrics:
         ttest_df = compare_models_with_ttest(
             baseline_results=(baseline_metrics, baseline_per_query, baseline_run),
             models_results=model_results_ttest,
             metrics=metrics,
             model_names=model_names_ttest
         )

         ttest_save_path = os.path.join(results_save_dir, 'ttest_results.txt')
         write_ttest_results(ttest_df, ttest_save_path, "Baseline")

         print("\nT-Test Results Summary (vs Baseline):")
         print(ttest_df)
    else:
        print("Skipping T-tests as baseline results are missing.")
    # ---------------------------------------------------------------


if __name__ == "__main__":
    # Recommend using CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Define the models and datasets to run
    run_models = ['distilbert-base-uncased', 'microsoft/MiniLM-L12-H384-uncased', 'distilroberta-base']
    run_datasets = ['fiqa', 'quora']

    metrics = [
        nDCG@100,
        R@100,
        RR
    ]

    print("Starting ensemble evaluation runs...")
    total_runs = len(run_models) * len(run_datasets)
    current_run = 0

    for model_name_run in run_models:
        for dataset_name_run in run_datasets:
            current_run += 1
            print(f"\n>>> Starting Run {current_run}/{total_runs} [Model: {model_name_run}, Dataset: {dataset_name_run}] <<<")
            try:
                run(model_name_run, dataset_name_run, metrics, device)
                print(f">>> Run {current_run}/{total_runs} Completed Successfully <<<")
            except Exception as e:
                print(f"!!! CRITICAL ERROR during run {current_run}/{total_runs} for {model_name_run} on {dataset_name_run}: {e} !!!")
                import traceback
                traceback.print_exc() # Print detailed traceback
                print("!!! Skipping to next run !!!")
            finally:
                # Explicit cleanup between major runs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\nAll ensemble evaluation runs completed.")