import sys

import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir import util
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from IRutils import dataprocessor
from IRutils.dataset import TripletRankingDataset, RankingDataset


def calculate_percentiles(query_lengths):
    # Calculate the percentiles
    t1 = np.percentile(query_lengths, 33)
    t2 = np.percentile(query_lengths, 67)
    return int(t1), int(t2)

def load(dataset_name):
    datasets = {'msmarco': ['dev', 'test'],
                'hotpotqa': ['train', 'dev', 'test'],
                'arguana': ['test'],
                'quora': ['dev', 'test'],
                'scidocs': ['test'],  # small
                'fever': ['train', 'dev', 'test'],  # large
                'climate-fever': ['test'],
                'scifact': ['train', 'test'],
                'fiqa': ['train', 'dev', 'test'],
                'nfcorpus': ['train', 'dev', 'test']
                }

    # Extract dataset from BEIR
    # Download and unzip the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets")

    train_available = False
    if 'train' in datasets[dataset_name]:
        # Load the dataset
        docs, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
        docs_test, queries_test, qrels_test = GenericDataLoader(data_folder=data_path).load(split="test")
        train_available = True
        print('Train and test set available!')
        return train_available, docs, queries, qrels, docs_test, queries_test, qrels_test

    else:
        # Load the dataset
        docs, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print('Only test set available!')

        return train_available, docs, queries, qrels, {}, {}, {}


def preprocess(queries, docs, qrels, model_name, length_setting, train_available, queries_test=None, qrels_test=None, max_len_doc=512, random_state=42, for_eval=False):
    # Initialize data processor and determine threshold
    # -----------------------------------------------------
    dp = dataprocessor.DataProcessor(queries, docs, qrels)

    query_lengths = [len(txt.split()) for txt in list(queries.values())]
    t1, t2 = calculate_percentiles(query_lengths)
    ranges = {'short': (1, t1), 'medium': (t1, t2), 'long': (t2, sys.maxsize), 'full': (1, sys.maxsize)}
    # -----------------------------------------------------

    # Extract test set
    # -----------------------------------------------------
    print(f'Dataset size: {len(queries)}')

    # first seperate the test set (include queries of all lengths)
    if not train_available:
        query_test, qrel_test = dp.get_testset(test_ratio=0.065, random_state=random_state)
        print(f'test size: {len(query_test)}')
    else:
        print(f'test size: {len(queries_test)}')
    # -----------------------------------------------------

    # Filter data by query length
    # -----------------------------------------------------
    full = True if length_setting == 'full' else False

    query_subset, qrels_subset = dp.get_subset(ranges[length_setting][0], ranges[length_setting][1],
                                               full=full)  # Adjust min/max length
    # -----------------------------------------------------

    # Split remaining data into train and val sets
    # -----------------------------------------------------
    print(f'Example query from {length_setting} subset:\n{query_subset.popitem()}')

    query_train, query_val, qrel_train, qrel_val = dp.train_val_split(train_ratio=0.8, val_ratio=0.2,
                                                                      random_state=random_state)  # adjust if needed

    print(f'Length of subset of {length_setting} validation queries: {len(query_val)}')
    print(f'Length of subset of {length_setting} training queries: {len(query_train)}')
    print(f'Length of subset of {length_setting} queries: {len(query_subset)}')
    # -----------------------------------------------------

    # Check if qrels contain negative samples
    # -----------------------------------------------------
    qrel_scores = list(qrels.values())
    relevance_scores = [list(item.values()) for item in qrel_scores]
    num_negatives = relevance_scores[0].count(0)
    print(f'Number of negatives in qrels: {num_negatives}')
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # -----------------------------------------------------

    # Create dataloaders
    # -----------------------------------------------------
    if not for_eval:
        print('Creating training dataset...')
        train_dataset = TripletRankingDataset(query_train, docs, qrel_train, tokenizer, num_negatives,
                                              max_length=max_len_doc)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    else:
        print('Skipping train set preparation...')
        train_loader = None

    print('Creating validation dataset...')
    val_dataset = TripletRankingDataset(query_val, docs, qrel_val, tokenizer, num_negatives, max_length=max_len_doc)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print('Creating test dataset...')
    if train_available:
        test_dataset = RankingDataset(queries_test, docs, qrels_test, tokenizer)  # query-doc instead of triplets
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader, {}, {}, query_val, qrel_val
    else:
        test_dataset = RankingDataset(query_test, docs, qrel_test, tokenizer)  # query-doc instead of triplets
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader, query_test, qrel_test, query_val, qrel_val
    # -----------------------------------------------------


