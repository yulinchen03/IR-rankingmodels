import os
import logging
import torch
from torch.optim import AdamW
from ir_measures import nDCG, P, R, RR
from IRutils import models, train
from IRutils.load_data import load, preprocess


def run(model_name, dataset_name, length_setting):
    logging.disable(logging.WARNING)

    max_len_doc = 512  # max token length
    random_state = 42

    train_available, docs, queries, qrels, docs_test, queries_test, qrels_test = load(dataset_name)
    print('Loading complete!')

    if train_available:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, _, _ = preprocess(queries, docs, qrels,
                                                                                                 model_name,
                                                                                                 length_setting,
                                                                                                 train_available,
                                                                                                 queries_test=queries_test,
                                                                                                 qrels_test=qrels_test,
                                                                                                 max_len_doc=max_len_doc,
                                                                                                 random_state=random_state)
    else:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, _, _ = preprocess(queries, docs, qrels,
                                                                                                 model_name,
                                                                                                 length_setting,
                                                                                                 train_available,
                                                                                                 max_len_doc=max_len_doc,
                                                                                                 random_state=random_state)

    print('Preprocessing complete!')

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = models.TripletRankerModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Define model save dir
    os.makedirs(f'models/{model_name}/{dataset_name}', exist_ok=True)
    model_path = os.path.join(os.getcwd(), f'models/{model_name}/{dataset_name}/{length_setting}_queries.pth')

    print(f'Checking model path for existing model...: {model_path}')
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train the model
        # model = train.train_triplet_ranker(model, train_loader, val_loader, optimizer, device, model_path)
        model = train.train_triplet_ranker_amp(model, train_loader, val_loader, optimizer, device, model_path) # amp enabled
    print('Run complete!')


if __name__ == "__main__":
    run_models = ['microsoft/MiniLM-L12-H384-uncased', 'distilbert-base-uncased', 'distilroberta-base-uncased']  # customize the model architectures you want to train
    run_datasets = ['fiqa', 'quora']  # customize the datasets you want your models to fine-tune on
    run_length_settings = ['short', 'medium', 'long', 'full']
    current_run = 0

    # Example for multiple runs (uncomment above lines and comment single run lines)
    print("Starting training runs...")
    total_runs = len(run_models) * len(run_datasets) * len(run_length_settings)

    for model in run_models:
        for dataset in run_datasets:
            for length in run_length_settings:
                current_run += 1
                print(f"\n>>> Starting Run {current_run}/{total_runs} <<<")
                try:
                    print(f'Now training model {model} on dataset {dataset} for {length} query lengths...')
                    run(model, dataset, length)
                except Exception as e:
                    print(f"!!! CRITICAL ERROR during run for {model} on {dataset}: {e} !!!")
                    print("!!! Skipping to next run !!!")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

