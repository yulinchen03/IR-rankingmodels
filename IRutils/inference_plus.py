import torch
import ir_measures
from ir_measures import *  # Common metrics like nDCG, RR, R, P
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from scipy import stats
import gc  # Import garbage collector
import os  # Import os for path joining
from transformers import AutoTokenizer  # Keep if tokenizer needed here, else remove


# Helper function to encode queries (assuming it's needed by evaluate functions)
# Ensure this function handles batching or single query encoding appropriately
def encode_queries(queries_dict, model, tokenizer, device, max_length=512):
    """Encodes a dictionary of queries."""
    model.eval()
    model.to(device)
    query_embeddings = {}
    # Consider batching if many queries
    with torch.no_grad():
        for qid, text in tqdm(queries_dict.items(), desc="Encoding Queries"):
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = model.get_embedding(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']).cpu().numpy()  # Assuming CLS token pooling
            query_embeddings[qid] = embeddings.flatten()  # Store as flat numpy array
    return query_embeddings


# Helper function for scoring (dot product)
def score_queries(query_embeddings, doc_embeddings, device):
    """Calculates scores between query embeddings and all document embeddings."""
    scores = {}
    doc_ids = list(doc_embeddings.keys())
    # Convert doc embeddings dict to a tensor matrix ONCE for efficiency
    # Ensure consistent ordering
    try:
        # Stack tensors directly if they are already tensors
        doc_matrix = torch.stack([doc_embeddings[doc_id] for doc_id in doc_ids]).to(device)  # Move to target device
    except TypeError:
        # If embeddings were loaded as numpy arrays, convert them
        doc_matrix = torch.tensor(np.stack([doc_embeddings[doc_id] for doc_id in doc_ids]), dtype=torch.float32).to(
            device)

    with torch.no_grad():
        for qid, q_emb in tqdm(query_embeddings.items(), desc="Scoring Queries"):
            q_tensor = torch.tensor(q_emb, dtype=torch.float32).to(device)  # Move query to device
            # Perform dot product scoring
            q_scores = torch.matmul(doc_matrix,
                                    q_tensor).cpu().numpy()  # Calculate scores on device, move result to CPU
            scores[qid] = {doc_ids[i]: float(q_scores[i]) for i in range(len(doc_ids))}

    # Cleanup GPU memory used by doc_matrix
    del doc_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------
# Modified Evaluation Functions
# ---------------------------------------------------------------------

def evaluate(model, tokenizer, queries, doc_embeddings_path, qrels, device, max_length=512, metrics=None):
    """Evaluates a single model using embeddings loaded from a path."""
    if metrics is None:
        metrics = [nDCG @ 100, R @ 100, RR]  # Default metrics

    print(f"Evaluating baseline model. Loading embeddings from: {doc_embeddings_path}")
    doc_embeddings = None
    scores = None
    try:
        # --- Load embeddings ---
        doc_embeddings = torch.load(doc_embeddings_path, map_location=device)
        print(f"Loaded {len(doc_embeddings)} document embeddings.")
        if doc_embeddings and device != 'cpu':  # Check device if not CPU
            first_key = next(iter(doc_embeddings))
            print(f"Document embeddings loaded onto device: {doc_embeddings[first_key].device}")

        # --- Encode Queries ---
        query_embeddings = encode_queries(queries, model, tokenizer, device, max_length)

        # --- Score Queries ---
        scores = score_queries(query_embeddings, doc_embeddings, device)

        # --- Evaluate with ir_measures ---
        evaluator = ir_measures.evaluator(metrics, qrels)
        results = evaluator.calc_aggregate(scores)  # Aggregate results are fine

        # Per-Query Dictionary Creation ---
        per_query_results_list = list(evaluator.iter_calc(scores))  # Collect results first
        per_query_dict = {}
        for res in per_query_results_list:
            qid = res.query_id
            metric_str = str(res.measure)  # Get the specific measure string from the result
            value = res.value

            # Build the nested dictionary correctly
            if qid not in per_query_dict:
                per_query_dict[qid] = {}
            # Assign the value only to the correct metric key
            per_query_dict[qid][metric_str] = value

        # Convert run format if needed (scores dict is usually compatible)
        run_df = pd.DataFrame([(q, d, s) for q, docs_scores in scores.items() for d, s in docs_scores.items()],
                              columns=['query_id', 'doc_id', 'score'])

        print("Baseline evaluation complete.")
        return results, per_query_dict, run_df

    except Exception as e:
        print(f"Error during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results on error? Or re-raise? Adjust as needed.
        return {}, {}, pd.DataFrame()
    finally:
        # --- Cleanup ---
        print("Cleaning up baseline evaluation resources...")
        del doc_embeddings
        del scores
        del query_embeddings  # Also clear query embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_average_ensemble(models, tokenizer, queries, embedding_save_dir, qrels, device, max_len_doc=512,
                              metrics=None):
    """Evaluates average ensemble, loading embeddings from a directory."""
    if metrics is None:
        metrics = [nDCG @ 100, R @ 100, RR]  # Default metrics

    print(f"Evaluating average ensemble. Loading embeddings from: {embedding_save_dir}")
    loaded_embeddings = {}
    scores = {}
    model_keys = list(models.keys())  # e.g., ['short', 'medium', 'long']

    try:
        # --- Load all necessary embeddings ---
        for key in model_keys:
            path = os.path.join(embedding_save_dir, f'{key}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embedding file not found for model '{key}' at {path}")
            loaded_embeddings[key] = torch.load(path, map_location=device)
            print(f"Loaded embeddings for {key} model.")
            if loaded_embeddings[key] and device != 'cpu':
                first_doc_id = next(iter(loaded_embeddings[key]))
                print(f"  Device: {loaded_embeddings[key][first_doc_id].device}")

        # --- Encode Queries (Average embedding across models) ---
        print("Encoding queries using average ensemble...")
        query_embeddings_avg = {}
        # Get query embeddings from each model
        all_query_embeddings = {}
        for key in model_keys:
            all_query_embeddings[key] = encode_queries(queries, models[key], tokenizer, device, max_len_doc)

        # Average query embeddings
        qids = list(queries.keys())
        for qid in tqdm(qids, desc="Averaging Query Embeddings"):
            avg_emb = np.mean([all_query_embeddings[key][qid] for key in model_keys], axis=0)
            query_embeddings_avg[qid] = avg_emb

        del all_query_embeddings  # Free memory from individual query embeddings
        gc.collect()

        # --- Score Queries (Average score across models) ---
        # This requires scoring against each model's doc embeddings separately and then averaging
        print("Scoring queries using average ensemble...")
        final_scores = {}
        doc_ids_list = list(loaded_embeddings[model_keys[0]].keys())  # Assume all have same doc ids

        # Pre-stack document embeddings for each model
        doc_matrices = {}
        for key in model_keys:
            try:
                doc_matrices[key] = torch.stack([loaded_embeddings[key][doc_id] for doc_id in doc_ids_list]).to(device)
            except TypeError:  # Handle numpy arrays
                doc_matrices[key] = torch.tensor(np.stack([loaded_embeddings[key][doc_id] for doc_id in doc_ids_list]),
                                                 dtype=torch.float32).to(device)

        with torch.no_grad():
            for qid, q_emb_avg in tqdm(query_embeddings_avg.items(), desc="Scoring Avg Ensemble"):
                q_tensor_avg = torch.tensor(q_emb_avg, dtype=torch.float32).to(device)
                avg_doc_scores = None

                for key in model_keys:
                    # Score query_avg against doc_embeddings[key]
                    q_scores_for_key = torch.matmul(doc_matrices[key], q_tensor_avg).cpu()  # Keep on CPU for averaging
                    if avg_doc_scores is None:
                        avg_doc_scores = q_scores_for_key
                    else:
                        avg_doc_scores += q_scores_for_key

                avg_doc_scores /= len(model_keys)
                final_scores[qid] = {doc_ids_list[i]: float(avg_doc_scores[i]) for i in range(len(doc_ids_list))}

        scores = final_scores
        del doc_matrices  # Free GPU memory for doc matrices
        del query_embeddings_avg
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Evaluate with ir_measures ---
        evaluator = ir_measures.evaluator(metrics, qrels)
        results = evaluator.calc_aggregate(scores)

        # Per-Query Dictionary Creation ---
        per_query_results_list = list(evaluator.iter_calc(scores))  # Collect results first
        per_query_dict = {}
        for res in per_query_results_list:
            qid = res.query_id
            metric_str = str(res.measure)  # Get the specific measure string from the result
            value = res.value

            # Build the nested dictionary correctly
            if qid not in per_query_dict:
                per_query_dict[qid] = {}
            # Assign the value only to the correct metric key
            per_query_dict[qid][metric_str] = value

        run_df = pd.DataFrame([(q, d, s) for q, docs_scores in scores.items() for d, s in docs_scores.items()],
                              columns=['query_id', 'doc_id', 'score'])

        print("Average ensemble evaluation complete.")
        return results, per_query_dict, run_df

    except Exception as e:
        print(f"Error during average ensemble evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}, pd.DataFrame()
    finally:
        # --- Cleanup ---
        print("Cleaning up average ensemble resources...")
        del loaded_embeddings
        del scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_conditional_ensemble(models, tokenizer, queries, embedding_save_dir, qrels, t1, t2, device,
                                  max_len_doc=512, metrics=None):
    """Evaluates conditional ensemble, loading embeddings as needed."""
    if metrics is None:
        metrics = [nDCG @ 100, R @ 100, RR]  # Default metrics

    print(f"Evaluating conditional ensemble. Loading embeddings from: {embedding_save_dir}")
    loaded_embeddings = {}  # Store embeddings as they are loaded
    scores = {}
    model_keys = ['short', 'medium', 'long']  # Expected keys

    try:
        # --- Pre-encode all queries with their respective models ONCE ---
        print("Pre-encoding queries for conditional ensemble...")
        query_embeddings = {}  # Dict: qid -> embedding
        qids = list(queries.keys())

        for qid in tqdm(qids, desc="Encoding Queries Conditionally"):
            query_text = queries[qid]
            query_length = len(query_text.split())

            if query_length <= t1:
                model_key = 'short'
            elif query_length <= t2:
                model_key = 'medium'
            else:
                model_key = 'long'

            # Encode using the chosen model
            model = models[model_key]
            with torch.no_grad():
                inputs = tokenizer(query_text, return_tensors='pt', max_length=max_len_doc, truncation=True,
                                   padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                q_emb = model.get_embedding(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']).cpu().numpy().flatten()
                query_embeddings[qid] = q_emb

        # --- Score Queries ---
        # We need to load the corresponding doc embeddings for each query group
        print("Scoring queries using conditional ensemble...")
        final_scores = {}
        doc_ids_list = None  # Will be determined when first embedding set is loaded

        # Group queries by the model they use
        grouped_queries = {'short': [], 'medium': [], 'long': []}
        for qid in qids:
            query_length = len(queries[qid].split())
            if query_length <= t1:
                grouped_queries['short'].append(qid)
            elif query_length <= t2:
                grouped_queries['medium'].append(qid)
            else:
                grouped_queries['long'].append(qid)

        # Process each group
        for model_key in model_keys:
            if not grouped_queries[model_key]: continue  # Skip if no queries for this group

            print(f"  Processing group: {model_key} ({len(grouped_queries[model_key])} queries)")
            # Load embeddings for this group
            path = os.path.join(embedding_save_dir, f'{model_key}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embedding file not found for model '{model_key}' at {path}")
            current_doc_embeddings = torch.load(path, map_location=device)
            print(f"  Loaded embeddings for {model_key} model.")
            if doc_ids_list is None:
                doc_ids_list = list(current_doc_embeddings.keys())  # Assuming all doc sets are the same

            # Create doc matrix for scoring
            try:
                doc_matrix = torch.stack([current_doc_embeddings[doc_id] for doc_id in doc_ids_list]).to(device)
            except TypeError:
                doc_matrix = torch.tensor(np.stack([current_doc_embeddings[doc_id] for doc_id in doc_ids_list]),
                                          dtype=torch.float32).to(device)

            # Score queries in this group
            with torch.no_grad():
                for qid in tqdm(grouped_queries[model_key], desc=f"Scoring {model_key} group"):
                    q_tensor = torch.tensor(query_embeddings[qid], dtype=torch.float32).to(device)
                    q_scores = torch.matmul(doc_matrix, q_tensor).cpu().numpy()
                    final_scores[qid] = {doc_ids_list[i]: float(q_scores[i]) for i in range(len(doc_ids_list))}

            # Clean up memory for this group's embeddings
            del current_doc_embeddings
            del doc_matrix
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        scores = final_scores
        del query_embeddings  # Clean up all query embeddings now
        gc.collect()

        # --- Evaluate with ir_measures ---
        evaluator = ir_measures.evaluator(metrics, qrels)
        results = evaluator.calc_aggregate(scores)

        # Per-Query Dictionary Creation ---
        per_query_results_list = list(evaluator.iter_calc(scores))  # Collect results first
        per_query_dict = {}
        for res in per_query_results_list:
            qid = res.query_id
            metric_str = str(res.measure)  # Get the specific measure string from the result
            value = res.value

            # Build the nested dictionary correctly
            if qid not in per_query_dict:
                per_query_dict[qid] = {}
            # Assign the value only to the correct metric key
            per_query_dict[qid][metric_str] = value

        run_df = pd.DataFrame([(q, d, s) for q, docs_scores in scores.items() for d, s in docs_scores.items()],
                              columns=['query_id', 'doc_id', 'score'])

        print("Conditional ensemble evaluation complete.")
        return results, per_query_dict, run_df

    except Exception as e:
        print(f"Error during conditional ensemble evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}, pd.DataFrame()
    finally:
        # --- Cleanup ---
        # Most cleanup now happens within the loop, but ensure loaded_embeddings is cleared if populated
        print("Cleaning up conditional ensemble resources...")
        del loaded_embeddings  # Should be empty if loop logic worked, but clear just in case
        del scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_weighted_average_ensemble(models, tokenizer, queries, embedding_save_dir, weights_config, qrels, t1, t2,
                                       device, max_len_doc=512, metrics=None):
    """Evaluates weighted average ensemble, loading embeddings from directory."""
    if metrics is None:
        metrics = [nDCG @ 100, R @ 100, RR]  # Default metrics

    print(f"Evaluating weighted average ensemble. Loading embeddings from: {embedding_save_dir}")
    loaded_embeddings = {}
    scores = {}
    model_keys = ['short', 'medium', 'long']  # Expected keys and order for weights

    try:
        # --- Load all necessary embeddings ---
        for key in model_keys:
            path = os.path.join(embedding_save_dir, f'{key}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embedding file not found for model '{key}' at {path}")
            loaded_embeddings[key] = torch.load(path, map_location=device)
            print(f"Loaded embeddings for {key} model.")
            if loaded_embeddings[key] and device != 'cpu':
                first_doc_id = next(iter(loaded_embeddings[key]))
                print(f"  Device: {loaded_embeddings[key][first_doc_id].device}")

        # --- Encode queries with ALL models (needed for weighted scoring) ---
        print("Pre-encoding queries with all ensemble models...")
        all_query_embeddings = {}  # Dict: model_key -> {qid -> embedding}
        for key in model_keys:
            all_query_embeddings[key] = encode_queries(queries, models[key], tokenizer, device, max_len_doc)

        # --- Score Queries (Weighted average score) ---
        print("Scoring queries using weighted average ensemble...")
        final_scores = {}
        doc_ids_list = list(loaded_embeddings[model_keys[0]].keys())  # Assume consistent doc IDs

        # Pre-stack document embeddings
        doc_matrices = {}
        for key in model_keys:
            try:
                doc_matrices[key] = torch.stack([loaded_embeddings[key][doc_id] for doc_id in doc_ids_list]).to(device)
            except TypeError:
                doc_matrices[key] = torch.tensor(np.stack([loaded_embeddings[key][doc_id] for doc_id in doc_ids_list]),
                                                 dtype=torch.float32).to(device)

        # Determine weights for each query
        query_weights = {}
        qids = list(queries.keys())
        for qid in qids:
            query_length = len(queries[qid].split())
            if query_length <= t1:
                weight_key = 'short'
            elif query_length <= t2:
                weight_key = 'medium'
            else:
                weight_key = 'long'
            query_weights[qid] = weights_config[weight_key]  # Get the list [w_short, w_medium, w_long]

        # Calculate weighted scores
        with torch.no_grad():
            for qid in tqdm(qids, desc="Scoring Weighted Ensemble"):
                weights = query_weights[qid]
                weighted_doc_scores = None

                for i, key in enumerate(model_keys):  # Order matters: short, medium, long
                    # Use the query embedding generated by THIS model (key)
                    q_emb = all_query_embeddings[key][qid]
                    q_tensor = torch.tensor(q_emb, dtype=torch.float32).to(device)

                    # Score query[key] against docs[key]
                    q_scores_for_key = torch.matmul(doc_matrices[key], q_tensor).cpu()  # Keep on CPU

                    if weighted_doc_scores is None:
                        weighted_doc_scores = q_scores_for_key * weights[i]
                    else:
                        weighted_doc_scores += q_scores_for_key * weights[i]

                # Note: Weights should sum to 1, otherwise normalization might be needed depending on interpretation
                final_scores[qid] = {doc_ids_list[j]: float(weighted_doc_scores[j]) for j in range(len(doc_ids_list))}

        scores = final_scores
        del doc_matrices
        del all_query_embeddings
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Evaluate with ir_measures ---
        evaluator = ir_measures.evaluator(metrics, qrels)
        results = evaluator.calc_aggregate(scores)

        # Per-Query Dictionary Creation ---
        per_query_results_list = list(evaluator.iter_calc(scores))  # Collect results first
        per_query_dict = {}
        for res in per_query_results_list:
            qid = res.query_id
            metric_str = str(res.measure)  # Get the specific measure string from the result
            value = res.value

            # Build the nested dictionary correctly
            if qid not in per_query_dict:
                per_query_dict[qid] = {}
            # Assign the value only to the correct metric key
            per_query_dict[qid][metric_str] = value

        run_df = pd.DataFrame([(q, d, s) for q, docs_scores in scores.items() for d, s in docs_scores.items()],
                              columns=['query_id', 'doc_id', 'score'])

        print("Weighted average ensemble evaluation complete.")
        return results, per_query_dict, run_df

    except Exception as e:
        print(f"Error during weighted average ensemble evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}, pd.DataFrame()
    finally:
        # --- Cleanup ---
        print("Cleaning up weighted average ensemble resources...")
        del loaded_embeddings
        del scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Utility Functions (Keep as they are)
# ---------------------------------------------------------------------

def perform_ttest(baseline_per_query, model_per_query, metric, alpha=0.05):
    """
    Performs a paired t-test between baseline and model results.

    Args:
        baseline_per_query: Per-query results from the baseline model
        model_per_query: Per-query results from the comparison model
        metric: Metric object to test (e.g., nDCG @ 10)
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, and significance result
    """
    # Extract per-query scores for the specified metric
    baseline_scores = []
    model_scores = []

    # Get common query IDs
    common_qids = set(baseline_per_query.keys()) & set(model_per_query.keys())

    for qid in common_qids:
        # Find the metric in baseline_per_query[qid]
        for entry, score in baseline_per_query[qid].items():
            if entry == str(metric):
                baseline_scores.append(score)

        # Find the metric in model_per_query[qid]
        for entry, score in model_per_query[qid].items():
            if entry == str(metric):
                model_scores.append(score)


    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'improvement': np.mean(model_scores) > np.mean(baseline_scores)
    }


def write_results(metrics_dict, save_path, model_name, dataset_name, length_setting):
    """Writes evaluation metrics to a file."""
    try:
        with open(save_path, 'w') as f:
            f.write(f"Results for Model: {model_name}, Dataset: {dataset_name}, Setting: {length_setting}\n")
            f.write("-" * 30 + "\n")
            if not metrics_dict:
                f.write("No metrics available.\n")
                return
            for metric, score in metrics_dict.items():
                # Ensure metric object is converted to string representation
                metric_str = str(metric) if isinstance(metric, ir_measures.Measure) else metric
                f.write(f"{metric_str}: {score:.4f}\n")
        print(f"Results successfully written to {save_path}")
    except Exception as e:
        print(f"Error writing results to {save_path}: {e}")


def compare_models_with_ttest(baseline_results, models_results, metrics, model_names):
    """
    Compares each model to the baseline using t-tests for multiple metrics.

    Args:
        baseline_results: Tuple with (aggregate_scores, per_query_scores, run) from baseline
        models_results: List of tuples with results for each model being compared
        metrics: List of metric objects to test
        model_names: List of names for each model

    Returns:
        DataFrame with comparison results
    """
    _, baseline_per_query, _ = baseline_results

    results = []

    for i, (_, model_per_query, _) in enumerate(models_results):
        model_name = model_names[i]

        for metric in metrics:
            test_result = perform_ttest(baseline_per_query, model_per_query, metric)

            # Get mean scores for baseline and this model
            baseline_mean = np.mean([baseline_per_query[qid][entry] for qid in baseline_per_query
                                     for entry in baseline_per_query[qid] if entry == str(metric)])
            model_mean = np.mean([model_per_query[qid][entry] for qid in model_per_query
                                  for entry in model_per_query[qid] if entry == str(metric)])

            results.append({
                'Model': model_name,
                'Metric': str(metric),
                'Baseline Score': baseline_mean,
                'Model Score': model_mean,
                'Difference': model_mean - baseline_mean,
                'Improvement %': ((model_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0,
                'T-Statistic': test_result['t_statistic'],
                'P-Value': test_result['p_value'],
                'Significant': test_result['significant'],
                'Improvement': test_result['improvement']
            })

    return pd.DataFrame(results)


def write_ttest_results(ttest_df, save_path, baseline_name="Baseline"):
    """Writes the t-test results DataFrame to a formatted text file."""
    try:
        with open(save_path, 'w') as f:
            f.write(f"T-Test Results (Comparison against {baseline_name})\n")
            f.write("=" * 50 + "\n\n")
            if ttest_df.empty:
                f.write("No t-test results available.\n")
                return

            # Format the DataFrame to string for better readability in the text file
            # Adjust max_colwidth and other options as needed
            df_string = ttest_df.to_string(
                index=False,
                float_format='{:.4f}'.format,  # Format floats nicely
                justify='left',  # Left-align columns
                col_space=12  # Add some space between columns
            )
            f.write(df_string)
        print(f"T-test results successfully written to {save_path}")
    except Exception as e:
        print(f"Error writing t-test results to {save_path}: {e}")