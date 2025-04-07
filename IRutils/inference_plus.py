import os
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F # Needed for cosine similarity
from ir_measures import calc_aggregate
from tqdm import tqdm
# from tqdm import tqdm
from ir_measures import *
from scipy import stats  # For t-test
import pandas as pd
from transformers import PreTrainedTokenizer

from IRutils.dataset import encode_query
from IRutils.models import TripletRankerModel


def write_results(metric_scores, save_path, model_name, dataset_name, length_setting):
    """
    Saves evaluation results to a file, reporting the specified metrics.

    Args:
        metric_scores: A dictionary-like object mapping ir_measures metric objects
                       to their calculated scores. Must include keys for the metrics
                       defined in `required_metrics`.
        save_path: Path to the file where results will be saved.
        model_name: Name of the model being evaluated.
        dataset_name: Name of the dataset used.
        length_setting: Description of the query length focus or model variant
                        (e.g., "short", "all", "baseline").
    """
    # Define the required metric objects for key access and reporting
    # These MUST match the metrics calculated in your evaluate function
    required_metrics = {
        'nDCG@100': nDCG @ 100,
        'RR': RR,
        'R@100': R @ 100,
    }

    # Check if all required metric scores are present in the input dict
    scores = {}
    missing_metrics = []
    found_metric_names = []
    for name, metric_obj in required_metrics.items():
        if metric_obj not in metric_scores:
            missing_metrics.append(name)
        else:
            scores[name] = metric_scores[metric_obj]
            found_metric_names.append(name)

    if missing_metrics:
        print(f"Metric scores not found for: {', '.join(missing_metrics)}. "
                     f"Metrics found were: {found_metric_names}. "
                     f"Ensure your evaluate function calculates all required metrics: "
                     f"{list(required_metrics.keys())}")
        # Decide if you want to proceed with partial results or exit
        # For now, we'll proceed with what we have, but log the error.
        # return # Uncomment this to exit if any required metric is missing

    print(f"Writing results for metrics: {list(scores.keys())} to {save_path}")

    # Save results to a file
    try:
        with open(save_path, "w") as f:
            f.write(f"Evaluation Results for {model_name} model ({length_setting}) on {dataset_name} dataset:\n")
            f.write("----------------------------------------------------\n")

            # --- Ranking/Overall Quality ---
            if 'nDCG@100' in scores:
                f.write(f"nDCG@100: {scores['nDCG@100']:.4f}  (Quality of Top 100 Ranking)\n")
            if 'RR' in scores:
                f.write(f"RR:  {scores['RR']:.4f}  (Mean Reciprocal Rank)\n")
            f.write("\n")

            # --- Recall Metric ---
            if 'R@100' in scores:
                f.write(f"R@100:   {scores['R@100']:.4f}  (Recall within Top 100)\n")
            f.write("\n")


            f.write("----------------------------------------------------\n")
            f.write("\n")
            f.write("Explanation of reported metrics:\n")
            f.write(
                "  nDCG@k: Measures ranking quality, rewarding highly relevant documents found earlier.\n"
                "          Normalized discount cumulative gain. Good overall top-k ranking indicator.\n"
            )
            f.write(
                "  RR:     The [Mean] Reciprocal Rank ([M]RR) is a precision-focused measure that scores "
                "          based on the reciprocal of the rank of the highest-scoring relevance document.\n"
            )
            f.write(
                "  R@k:    Recall@k. Fraction of *known* relevant documents found in the top k results.\n"
                "          Measures coverage within a practical top set (@100).\n"
            )

        print(f'Successfully written results to {save_path}.')
    except IOError as e:
        print(f"Failed to write results to {save_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during result writing: {e}")


def get_per_query_metrics(metrics, qrels, run):
    """Calculates metrics for each query individually."""
    per_query_results = {} # {qid: {metric_name: score}}
    # ir_measures.iter_calc returns an iterator of measurement objects
    # Each measurement has qid, metric, value
    for measurement in iter_calc(metrics, qrels, run):
        qid = measurement.query_id
        metric_name = str(measurement.measure) # Get a string representation
        score = measurement.value
        if qid not in per_query_results:
            per_query_results[qid] = {}
        per_query_results[qid][metric_name] = score
    return per_query_results


def evaluate(
    model: 'TripletRankerModel', # Your trained model instance
    tokenizer: 'PreTrainedTokenizer', # The model's tokenizer
    queries: Dict[str, str],          # Dict: qid -> query_text
    corpus_embeddings: Dict[str, torch.Tensor], # Dict: doc_id -> embedding_tensor (PRE-COMPUTED)
    qrels: Dict[str, Dict[str, int]], # Ground truth: qid -> {doc_id: relevance}
    device: torch.device,
    max_length: int,                  # Max sequence length for tokenizer
    score_measure: str = 'neg_l2'     # 'neg_l2', 'cos', or 'dot'
) -> (Dict, Dict, Dict):
    """
    Evaluates the model by scoring all documents against each query using pre-computed embeddings.

    Args:
        model: The trained ranking model (used for encoding queries).
        tokenizer: The tokenizer associated with the model.
        queries: Dictionary mapping query IDs to query text.
        corpus_embeddings: Dictionary mapping document IDs to their pre-computed embeddings (torch.Tensor).
                           These tensors should ideally be on CPU for efficient stacking first.
        qrels: Ground truth relevance judgements.
        device: The PyTorch device to run computations on (e.g., 'cuda:0' or 'cpu').
        max_length: Maximum sequence length for query tokenization.
        score_measure: The method to calculate query-document scores:
                       'neg_l2': Negative L2 norm (closer embeddings get higher scores).
                       'cos': Cosine similarity.
                       'dot': Dot product.

    Returns:
        A tuple containing:
        - aggregate_metric_scores (Dict): Dictionary of aggregate metric scores.
        - per_query_metric_scores (Dict): Dictionary of per-query metric scores.
        - run (Dict): The full run results (qid -> {doc_id: score}).
    """
    model.eval()
    model.to(device) # Ensure model is on the correct device for query encoding
    run = {}  # Format: {qid: {doc_id: score}}

    print(f"Preparing corpus embeddings tensor on device: {device}")
    doc_ids = list(corpus_embeddings.keys())
    # Stack embeddings (might happen on CPU if input tensors are CPU)
    all_doc_embeddings_stacked = torch.stack(list(corpus_embeddings.values()))
    # Move the entire stacked tensor to the target device *once*
    all_doc_embeddings = all_doc_embeddings_stacked.to(device)
    print(f"Corpus embeddings tensor shape: {all_doc_embeddings.shape}, Device: {all_doc_embeddings.device}")

    # Determine which queries to evaluate (those present in qrels and queries dict)
    evaluation_qids = [qid for qid in qrels.keys() if qid in queries]
    if len(evaluation_qids) != len(qrels):
         print(f"Evaluating on {len(evaluation_qids)} queries present in both qrels and the provided queries dict, out of {len(qrels)} total qrels.")
    if not evaluation_qids:
        print("No queries found that are present in both qrels and the queries dictionary. Cannot evaluate.")
        return {}, {}, {}

    print(f"Starting evaluation for {len(evaluation_qids)} queries using '{score_measure}' scoring...")
    with torch.no_grad():
        for qid in tqdm(evaluation_qids, desc="Evaluating Queries"):
            query_text = queries[qid]

            # 1. Encode the current query
            # Assuming encode_query returns embedding on the specified 'device', shape [1, hidden_size]
            query_embedding_batched = encode_query(query_text, model, tokenizer, max_length, device)
            query_embedding = query_embedding_batched.squeeze(0) # Shape: [hidden_size]

            # Ensure query embedding is on the correct device (should be already by encode_query)
            if query_embedding.device != device:
                 query_embedding = query_embedding.to(device)

            # 2. Calculate scores between the query and ALL document embeddings
            if score_measure == 'neg_l2':
                # Calculate squared L2 distance, then negate
                # Broadcasting query_embedding: [1, hidden_size] vs [N, hidden_size]
                distances = torch.norm(query_embedding.unsqueeze(0) - all_doc_embeddings, p=2, dim=1)
                scores = -distances
            elif score_measure == 'cos':
                # Compute cosine similarity
                scores = F.cosine_similarity(query_embedding.unsqueeze(0), all_doc_embeddings)
            elif score_measure == 'dot':
                # Compute dot product
                # Ensure query_embedding is shape (1, EmbDim) for matmul
                scores = torch.matmul(query_embedding.unsqueeze(0), all_doc_embeddings.T).squeeze(0)
            else:
                raise ValueError(f"Unknown score_measure: {score_measure}. Choose 'neg_l2', 'cos', or 'dot'.")

            # Move scores to CPU for storing in the run dictionary
            scores_list = scores.cpu().tolist()

            # 3. Store scores in the run dictionary
            run[qid] = {str(doc_id): score for doc_id, score in zip(doc_ids, scores_list)}


    print("Evaluation scoring complete. Calculating metrics...")

    # 4. Calculate metrics using the generated run and ground truth qrels
    # Define metrics (ensure these match what your evaluation library expects)
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]

    # Calculate aggregate metrics
    aggregate_scores = calc_aggregate(metrics, qrels, run)

    print("Calculating per-query metrics...")
    # Calculate per-query metrics
    per_query_scores = get_per_query_metrics(metrics, qrels, run)

    print("Metrics calculation complete.")

    return aggregate_scores, per_query_scores, run


def evaluate_average_ensemble(
    models: Dict[str, 'TripletRankerModel'], # Dict: model_name -> model_instance
    tokenizer: 'PreTrainedTokenizer',
    queries: Dict[str, str],
    corpus_embeddings_sets: Dict[str, Dict[str, torch.Tensor]], # Dict: model_name -> {doc_id: embedding}
    qrels: Dict[str, Dict[str, int]],
    device: torch.device,
    max_length: int,
    score_measure: str = 'neg_l2'
) -> Tuple[Dict[Measure, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Evaluates an ensemble by averaging scores from multiple models.
    Uses pre-computed document embeddings for each model.

    Args:
        models: Dictionary mapping model names to trained model instances.
        tokenizer: The tokenizer (assumed shared).
        queries: Dictionary mapping query IDs to query text.
        corpus_embeddings_sets: Dictionary mapping model names to their respective
                                pre-computed corpus embeddings ({doc_id: tensor}).
        qrels: Ground truth relevance judgements.
        device: PyTorch device.
        max_length: Max sequence length for query tokenization.
        score_measure: Scoring method ('neg_l2', 'cos', 'dot').

    Returns:
        Tuple: (aggregate_metric_scores, per_query_metric_scores, run)
    """
    run = {}
    model_names = list(models.keys())
    if not model_names:
        print("Error: No models provided for ensemble evaluation.")
        return {}, {}, {}

    print(f"Evaluating Average Ensemble with models: {model_names}")

    # --- Prepare models and document embeddings ---
    stacked_doc_embeddings = {}
    doc_ids = None
    for name, model in models.items():
        if name == 'full':
            continue
        model.eval()
        model.to(device)
        print(f"Preparing embeddings for model: {name}")
        if name not in corpus_embeddings_sets:
            print(f"Error: Corpus embeddings not found for model '{name}'. Skipping.")
            continue # Or raise error

        current_doc_embeddings = corpus_embeddings_sets[name]
        if not current_doc_embeddings:
             print(f"Warning: Corpus embeddings for model '{name}' is empty.")
             continue

        current_doc_ids = list(current_doc_embeddings.keys())
        if doc_ids is None:
            doc_ids = current_doc_ids
        elif set(doc_ids) != set(current_doc_ids):
            print(f"Warning: Document ID mismatch between models. Using intersection.")
            # Potentially recalculate doc_ids based on intersection if strictness needed
            # For now, assume evaluation requires consistent doc sets or handle missing scores later
            pass # Sticking with the first set's doc_ids for simplicity, check later

        try:
            # Ensure embeddings are loaded in the correct doc_id order
            embeddings_list = [current_doc_embeddings[doc_id] for doc_id in doc_ids]
            stacked = torch.stack(embeddings_list).to(device)
            stacked_doc_embeddings[name] = stacked
            print(f"  Stacked embeddings shape: {stacked.shape}, Device: {stacked.device}")
        except KeyError as e:
             print(f"Error: Doc ID {e} not found in embeddings for model '{name}' while trying to align order. Check consistency.")
             return {}, {}, {} # Cannot proceed if alignment fails
        except Exception as e:
            print(f"Error stacking embeddings for model '{name}': {e}")
            return {}, {}, {}

    if not stacked_doc_embeddings or doc_ids is None:
         print("Error: Could not prepare document embeddings for any model.")
         return {}, {}, {}

    num_docs = len(doc_ids)
    active_model_names = list(stacked_doc_embeddings.keys()) # Models for which embeddings are ready

    # --- Evaluate Queries ---
    evaluation_qids = [qid for qid in qrels.keys() if qid in queries]
    if not evaluation_qids:
        print("Error: No queries found in both qrels and queries dict.")
        return {}, {}, {}
    print(f"Starting evaluation for {len(evaluation_qids)} queries using '{score_measure}' scoring...")

    with torch.no_grad():
        for qid in tqdm(evaluation_qids, desc="Evaluating Queries (Avg Ensemble)"):
            query_text = queries[qid]
            model_scores_list = [] # Store score tensor [num_docs] from each model

            for name in active_model_names:
                model = models[name]
                doc_embeddings = stacked_doc_embeddings[name] # [num_docs, hidden_size]

                # 1. Encode query with the current model
                try:
                    query_embedding = encode_query(query_text, model, tokenizer, max_length, device).squeeze(0) # [hidden_size]
                except Exception as e:
                    print(f"Error encoding query {qid} with model {name}: {e}. Skipping model for this query.")
                    continue # Skip this model's contribution

                if query_embedding.device != device:
                    query_embedding = query_embedding.to(device)

                # 2. Calculate scores against all docs for this model
                try:
                    if score_measure == 'neg_l2':
                        distances = torch.norm(query_embedding.unsqueeze(0) - doc_embeddings, p=2, dim=1)
                        scores = -distances
                    elif score_measure == 'cos':
                        scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings, dim=1)
                    elif score_measure == 'dot':
                        scores = torch.matmul(doc_embeddings, query_embedding)
                    else:
                        raise ValueError(f"Invalid score_measure: {score_measure}")
                    model_scores_list.append(scores) # Append tensor of shape [num_docs]
                except Exception as e:
                     print(f"Error calculating scores for qid {qid} with model {name}: {e}. Skipping model for this query.")
                     continue # Skip this model's contribution

            # 3. Average scores across models
            if not model_scores_list:
                print(f"Warning: No scores could be computed for query {qid}. Assigning empty results.")
                run[qid] = {}
                continue

            # Stack scores [num_models, num_docs] and average
            final_scores = torch.stack(model_scores_list).mean(dim=0) # [num_docs]
            scores_list_cpu = final_scores.cpu().tolist()

            # 4. Store final averaged scores
            run[qid] = {str(doc_id): score for doc_id, score in zip(doc_ids, scores_list_cpu)}

    print("Scoring complete. Calculating metrics...")
    # --- Calculate Metrics ---
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]
    valid_qids_for_metrics = {qid for qid in qrels if qid in run and run[qid]}
    filtered_qrels = {qid: qrels[qid] for qid in valid_qids_for_metrics}
    filtered_run = {qid: run[qid] for qid in valid_qids_for_metrics}

    if not filtered_run:
         print("Warning: No valid queries with scores found to calculate metrics for average ensemble.")
         return {}, {}, run

    aggregate_scores = calc_aggregate(metrics, filtered_qrels, filtered_run)

    print("Calculating per-query metrics...")
    per_query_scores = get_per_query_metrics(metrics, filtered_qrels, filtered_run)
    print("Metrics calculation complete.")

    return aggregate_scores, per_query_scores, run



def evaluate_conditional_ensemble(
    models: Dict[str, 'TripletRankerModel'], # Should contain 'short', 'medium', 'long' keys
    tokenizer: 'PreTrainedTokenizer',
    queries: Dict[str, str],
    corpus_embeddings_sets: Dict[str, Dict[str, torch.Tensor]], # Keys must match model keys
    qrels: Dict[str, Dict[str, int]],
    t1: int, # Threshold 1 for query length
    t2: int, # Threshold 2 for query length
    device: torch.device,
    max_length: int,
    score_measure: str = 'neg_l2',
    queries_test: Dict[str, str] = None # Keep for query text lookup (can be same as queries)
) -> Tuple[Dict[Measure, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Evaluates a conditional ensemble based on query length.
    Uses pre-computed document embeddings for each model.

    Args:
        models: Dict mapping 'short', 'medium', 'long' to model instances.
        tokenizer: The tokenizer.
        queries: Dict mapping query IDs to query text (used for evaluation loop).
        corpus_embeddings_sets: Dict mapping 'short', 'medium', 'long' to their
                                respective pre-computed corpus embeddings.
        qrels: Ground truth relevance judgements.
        t1, t2: Query length thresholds (words).
        device: PyTorch device.
        max_length: Max sequence length for tokenization.
        score_measure: Scoring method ('neg_l2', 'cos', 'dot').
        queries_test: Dict mapping qid to query text (used for length check). If None, uses `queries`.

    Returns:
        Tuple: (aggregate_metric_scores, per_query_metric_scores, run)
    """
    run = {}
    required_keys = ['short', 'medium', 'long']
    if not all(key in models for key in required_keys):
        raise ValueError(f"Models dictionary must contain keys: {required_keys}")
    if not all(key in corpus_embeddings_sets for key in required_keys):
        raise ValueError(f"Corpus embeddings sets dictionary must contain keys: {required_keys}")

    if queries_test is None:
        queries_test = queries # Use the main queries dict if specific test dict not provided

    print(f"Evaluating Conditional Ensemble (Thresholds: {t1}, {t2})")

    # --- Prepare models and document embeddings ---
    stacked_doc_embeddings = {}
    doc_ids = None
    for name in required_keys:
        if name == 'full':
            continue
        model = models[name]
        model.eval()
        model.to(device)
        print(f"Preparing embeddings for model: {name}")
        current_doc_embeddings = corpus_embeddings_sets[name]
        if not current_doc_embeddings:
             print(f"Error: Corpus embeddings for model '{name}' is empty.")
             return {}, {}, {}

        current_doc_ids = list(current_doc_embeddings.keys())
        if doc_ids is None:
            doc_ids = current_doc_ids
        elif set(doc_ids) != set(current_doc_ids):
            print(f"Warning: Document ID mismatch between models. Using first set's IDs ('{required_keys[0]}').")
            # In a conditional setup, using different doc sets might be acceptable if intended,
            # but usually they should be consistent. Let's assume consistency for now.
            pass

        try:
            # Ensure embeddings are loaded in the correct doc_id order
            embeddings_list = [current_doc_embeddings[doc_id] for doc_id in doc_ids]
            stacked = torch.stack(embeddings_list).to(device)
            stacked_doc_embeddings[name] = stacked
            print(f"  Stacked embeddings shape: {stacked.shape}, Device: {stacked.device}")
        except KeyError as e:
             print(f"Error: Doc ID {e} not found in embeddings for model '{name}'. Check consistency.")
             return {}, {}, {}
        except Exception as e:
            print(f"Error stacking embeddings for model '{name}': {e}")
            return {}, {}, {}

    if not stacked_doc_embeddings or doc_ids is None:
         print("Error: Could not prepare document embeddings.")
         return {}, {}, {}

    num_docs = len(doc_ids)

    # --- Evaluate Queries ---
    evaluation_qids = [qid for qid in qrels.keys() if qid in queries and qid in queries_test]
    if not evaluation_qids:
        print("Error: No queries found in qrels, queries, and queries_test dicts.")
        return {}, {}, {}
    print(f"Starting evaluation for {len(evaluation_qids)} queries using '{score_measure}' scoring...")

    with torch.no_grad():
        for qid in tqdm(evaluation_qids, desc="Evaluating Queries (Cond. Ensemble)"):
            query_text_for_length = queries_test[qid]
            query_text_for_encoding = queries[qid]
            query_length = len(query_text_for_length.split())

            # 1. Select model based on query length
            if query_length <= t1:
                selected_model_name = 'short'
            elif t1 < query_length <= t2:
                selected_model_name = 'medium'
            else:  # query_length > t2
                selected_model_name = 'long'

            model = models[selected_model_name]
            doc_embeddings = stacked_doc_embeddings[selected_model_name]

            # 2. Encode query with the selected model
            try:
                query_embedding = encode_query(query_text_for_encoding, model, tokenizer, max_length, device).squeeze(0)
            except Exception as e:
                print(f"Error encoding query {qid} with selected model {selected_model_name}: {e}. Skipping query.")
                run[qid] = {}
                continue

            if query_embedding.device != device:
                query_embedding = query_embedding.to(device)

            # 3. Calculate scores against all docs using the selected model
            try:
                if score_measure == 'neg_l2':
                    distances = torch.norm(query_embedding.unsqueeze(0) - doc_embeddings, p=2, dim=1)
                    scores = -distances
                elif score_measure == 'cos':
                    scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings, dim=1)
                elif score_measure == 'dot':
                    scores = torch.matmul(doc_embeddings, query_embedding)
                else:
                    raise ValueError(f"Invalid score_measure: {score_measure}")

                scores_list_cpu = scores.cpu().tolist()

                 # 4. Store scores
                run[qid] = {str(doc_id): score for doc_id, score in zip(doc_ids, scores_list_cpu)}

            except Exception as e:
                 print(f"Error calculating scores for qid {qid} with model {selected_model_name}: {e}. Skipping query.")
                 run[qid] = {}


    print("Scoring complete. Calculating metrics...")
    # --- Calculate Metrics ---
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]
    valid_qids_for_metrics = {qid for qid in qrels if qid in run and run[qid]}
    filtered_qrels = {qid: qrels[qid] for qid in valid_qids_for_metrics}
    filtered_run = {qid: run[qid] for qid in valid_qids_for_metrics}

    if not filtered_run:
         print("Warning: No valid queries with scores found to calculate metrics for conditional ensemble.")
         return {}, {}, run

    aggregate_scores = calc_aggregate(metrics, filtered_qrels, filtered_run)

    print("Calculating per-query metrics...")
    per_query_scores = get_per_query_metrics(metrics, filtered_qrels, filtered_run)
    print("Metrics calculation complete.")

    return aggregate_scores, per_query_scores, run


def evaluate_weighted_average_ensemble(
    models: Dict[str, 'TripletRankerModel'], # Expects 'short', 'medium', 'long' keys
    tokenizer: 'PreTrainedTokenizer',
    queries: Dict[str, str],
    corpus_embeddings_sets: Dict[str, Dict[str, torch.Tensor]], # Keys must match model keys
    weights_config: Dict[str, List[float]], # e.g., {'short': [w1, w2, w3], 'medium': [...], 'long': [...]}
    qrels: Dict[str, Dict[str, int]],
    t1: int,
    t2: int,
    device: torch.device,
    max_length: int,
    score_measure: str = 'neg_l2',
    queries_test: Dict[str, str] = None # For query length check
) -> Tuple[Dict[Measure, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Evaluates an ensemble using weighted averaging based on query length.
    Uses pre-computed document embeddings for each model.

    Args:
        models: Dict mapping 'short', 'medium', 'long' to model instances.
        tokenizer: The tokenizer.
        queries: Dict mapping query IDs to query text (for evaluation loop).
        corpus_embeddings_sets: Dict mapping 'short', 'medium', 'long' to their embeddings.
        weights_config: Dict defining weights per query category.
                        Keys: 'short', 'medium', 'long'.
                        Values: List of weights [w_short, w_medium, w_long] for models['short'], models['medium'], models['long'].
        qrels: Ground truth.
        t1, t2: Query length thresholds.
        device: PyTorch device.
        max_length: Max sequence length.
        score_measure: Scoring method.
        queries_test: Dict mapping qid to query text (for length check). If None, uses `queries`.

    Returns:
        Tuple: (aggregate_metric_scores, per_query_metric_scores, run)
    """
    run = {}
    model_keys = ['short', 'medium', 'long'] # Expected order for weights
    if not all(key in models for key in model_keys):
        raise ValueError(f"Models dictionary must contain keys: {model_keys}")
    if not all(key in corpus_embeddings_sets for key in model_keys):
        raise ValueError(f"Corpus embeddings sets dictionary must contain keys: {model_keys}")
    if not all(key in weights_config for key in model_keys):
         raise ValueError(f"Weights config dictionary must contain keys: {model_keys}")

    if queries_test is None:
        queries_test = queries

    print(f"Evaluating Weighted Average Ensemble (Thresholds: {t1}, {t2})")

    # --- Prepare models and document embeddings ---
    stacked_doc_embeddings = {}
    doc_ids = None
    for name in model_keys:
        model = models[name]
        model.eval()
        model.to(device)
        print(f"Preparing embeddings for model: {name}")
        current_doc_embeddings = corpus_embeddings_sets[name]
        if not current_doc_embeddings:
             print(f"Error: Corpus embeddings for model '{name}' is empty.")
             return {}, {}, {}

        current_doc_ids = list(current_doc_embeddings.keys())
        if doc_ids is None:
            doc_ids = current_doc_ids
        elif set(doc_ids) != set(current_doc_ids):
            print(f"Warning: Document ID mismatch between models. Using first set's IDs ('{model_keys[0]}').")
            pass # Assume consistency

        try:
            embeddings_list = [current_doc_embeddings[doc_id] for doc_id in doc_ids]
            stacked = torch.stack(embeddings_list).to(device)
            stacked_doc_embeddings[name] = stacked
            print(f"  Stacked embeddings shape: {stacked.shape}, Device: {stacked.device}")
        except KeyError as e:
             print(f"Error: Doc ID {e} not found in embeddings for model '{name}'. Check consistency.")
             return {}, {}, {}
        except Exception as e:
            print(f"Error stacking embeddings for model '{name}': {e}")
            return {}, {}, {}

    if not stacked_doc_embeddings or doc_ids is None:
         print("Error: Could not prepare document embeddings.")
         return {}, {}, {}

    num_docs = len(doc_ids)
    num_models = len(model_keys)

    # --- Evaluate Queries ---
    evaluation_qids = [qid for qid in qrels.keys() if qid in queries and qid in queries_test]
    if not evaluation_qids:
        print("Error: No queries found in qrels, queries, and queries_test dicts.")
        return {}, {}, {}
    print(f"Starting evaluation for {len(evaluation_qids)} queries using '{score_measure}' scoring...")

    with torch.no_grad():
        for qid in tqdm(evaluation_qids, desc="Evaluating Queries (W.Avg Ensemble)"):
            query_text_for_length = queries_test[qid]
            query_text_for_encoding = queries[qid]
            query_length = len(query_text_for_length.split())

            # 1. Determine weights based on query length category
            if query_length <= t1:
                current_weights = weights_config['short']
            elif t1 < query_length <= t2:
                current_weights = weights_config['medium']
            else:  # query_length > t2
                current_weights = weights_config['long']

            if len(current_weights) != num_models:
                raise ValueError(f"Number of weights ({len(current_weights)}) for category "
                                 f"does not match number of models ({num_models}). "
                                 f"Weights should correspond to [{', '.join(model_keys)}].")

            # Normalize weights to sum to 1 (optional, but good practice)
            weight_sum = sum(current_weights)
            if weight_sum <= 0:
                print(f"Warning: Weights sum to {weight_sum} for query {qid}. Using equal weights.")
                normalized_weights = [1.0 / num_models] * num_models
            else:
                normalized_weights = [w / weight_sum for w in current_weights]


            # 2. Get scores from ALL models
            all_model_scores = [] # List to hold score tensors [num_docs]
            valid_model_indices_for_weighting = [] # Track which models successfully produced scores

            for idx, name in enumerate(model_keys):
                model = models[name]
                doc_embeddings = stacked_doc_embeddings[name]

                try:
                    query_embedding = encode_query(query_text_for_encoding, model, tokenizer, max_length, device).squeeze(0)
                    if query_embedding.device != device:
                        query_embedding = query_embedding.to(device)

                    if score_measure == 'neg_l2':
                        distances = torch.norm(query_embedding.unsqueeze(0) - doc_embeddings, p=2, dim=1)
                        scores = -distances
                    elif score_measure == 'cos':
                        scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings, dim=1)
                    elif score_measure == 'dot':
                        scores = torch.matmul(doc_embeddings, query_embedding)
                    else:
                        raise ValueError(f"Invalid score_measure: {score_measure}")

                    all_model_scores.append(scores)
                    valid_model_indices_for_weighting.append(idx) # Store original index

                except Exception as e:
                    print(f"Error scoring query {qid} with model {name}: {e}. Skipping model for this query's weighting.")
                    # We need a placeholder or adjust weights later if a model fails
                    all_model_scores.append(None) # Add None to keep list length consistent temporarily


            # 3. Calculate weighted average score
            if not valid_model_indices_for_weighting:
                 print(f"Warning: No scores could be computed for query {qid}. Assigning empty results.")
                 run[qid] = {}
                 continue

            final_scores = torch.zeros(num_docs, device=device)
            effective_weight_sum = 0.0 # Recalculate sum based on models that worked

            for idx in valid_model_indices_for_weighting:
                effective_weight_sum += normalized_weights[idx]

            if effective_weight_sum <= 0:
                 print(f"Warning: Effective weight sum is zero for query {qid} after model failures. Using equal weighting for successful models.")
                 num_successful = len(valid_model_indices_for_weighting)
                 for i, idx in enumerate(valid_model_indices_for_weighting):
                      if all_model_scores[idx] is not None:
                          final_scores += (1.0 / num_successful) * all_model_scores[idx]
            else:
                # Renormalize weights based on successful models
                for i, idx in enumerate(valid_model_indices_for_weighting):
                     if all_model_scores[idx] is not None:
                         weight = normalized_weights[idx] / effective_weight_sum
                         final_scores += weight * all_model_scores[idx]


            scores_list_cpu = final_scores.cpu().tolist()

            # 4. Store final weighted scores
            run[qid] = {str(doc_id): score for doc_id, score in zip(doc_ids, scores_list_cpu)}


    print("Scoring complete. Calculating metrics...")
    # --- Calculate Metrics ---
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]
    valid_qids_for_metrics = {qid for qid in qrels if qid in run and run[qid]}
    filtered_qrels = {qid: qrels[qid] for qid in valid_qids_for_metrics}
    filtered_run = {qid: run[qid] for qid in valid_qids_for_metrics}

    if not filtered_run:
         print("Warning: No valid queries with scores found to calculate metrics for weighted ensemble.")
         return {}, {}, run

    aggregate_scores = calc_aggregate(metrics, filtered_qrels, filtered_run)

    print("Calculating per-query metrics...")
    per_query_scores = get_per_query_metrics(metrics, filtered_qrels, filtered_run)
    print("Metrics calculation complete.")

    return aggregate_scores, per_query_scores, run


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
    """
    Saves t-test comparison results to a file.

    Args:
        ttest_df: DataFrame with t-test results
        save_path: Path to save the results
        baseline_name: Name of the baseline model for the report
    """
    with open(save_path, "w") as f:
        # --- Step 1: Define the title and output filename ---
        title = "T-Test Results Summary:"

        # --- Step 2: Convert DataFrame to string ---
        # df.to_string() creates a string representation matching console output
        df_string = ttest_df.to_string()

        # --- Step 3: Write to the text file ---
        try:
            with open(save_path, 'w') as f:
                f.write(title + "\n\n")  # Write the title and add a blank line
                f.write(df_string)  # Write the formatted DataFrame string
                f.write("\n")  # Add a final newline (optional)
            print(f"Successfully wrote DataFrame to {save_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    print(f"T-test results written to {save_path}")