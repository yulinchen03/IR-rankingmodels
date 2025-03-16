import pyterrier as pt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import pandas as pd
import shutil
import json
from sklearn.model_selection import train_test_split


class QueryLengthClassifier:
    """Configuration for query length classification"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            short_max: Maximum number of terms for short queries
            medium_max: Maximum number of terms for medium queries
        """
        self.config = config

    def classify_query(self, query: str) -> str:
        """Classify query based on number of terms"""
        terms = query.split()
        if len(terms) <= self.config.get("short_max"):
            return "short"
        elif len(terms) <= self.config.get("medium_max"):
            return "medium"
        else:
            return "long"


class DataManager:
    """Handles dataset loading and preprocessing"""

    def __init__(self, config: Dict[str, Any], classifier: QueryLengthClassifier):
        self.config = config
        self.classifier = classifier

    def load_dataset(self) -> Dict[str, Any]:
        """Load and prepare a dataset using PyTerrier"""
        dataset_name = self.config.get("dataset")
        print(f"Loading dataset: {dataset_name}")

        if dataset_name == "vaswani":
            dataset = pt.get_dataset("vaswani")
            index_variant = "terrier_stemmed"
        elif dataset_name == "msmarco":
            # For msmarco, we need to specify the index variant
            dataset = pt.get_dataset("msmarco_passage", variant="terrier_stemmed")
            index_variant = "terrier_stemmed"
            # For topics, specify a variant - 'dev' is good for experimentation
            topics_variant = self.config.get("topics_variant", "dev.small")

            print(f"Using MSMARCO topics variant: {topics_variant}")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # Get topics and qrels
        print('Fetching topics...')
        topics = dataset.get_topics(variant=topics_variant)
        print('Fetching qrels...')
        qrels = dataset.get_qrels(variant=topics_variant)

        # Classify queries by length
        print('Classifying query length...')
        topics['query_length_category'] = topics['query'].apply(
            self.classifier.classify_query
        )

        # Get corpus
        print('Fetching corpus...')
        corpus = dataset.get_corpus()

        print('Processing data...')
        # Split into train, validation and test sets (64%, 16%, 20%)
        train_topics, test_topics = train_test_split(
            topics, test_size=self.config.get("test_size", 0.2), random_state=42
        )
        train_topics, val_topics = train_test_split(
            train_topics, test_size=self.config.get("val_size", 0.2), random_state=42
        )

        # Filter qrels based on the qids in each split using column filtering
        train_qrels = qrels[qrels['qid'].isin(train_topics['qid'])]
        val_qrels = qrels[qrels['qid'].isin(val_topics['qid'])]
        test_qrels = qrels[qrels['qid'].isin(test_topics['qid'])]

        # Group training data by query length
        train_by_length = {
            length: train_topics[train_topics['query_length_category'] == length]
            for length in ['short', 'medium', 'long']
        }

        # Grouping qrel training data by length
        train_qrels_by_length = {
            length: train_qrels[train_qrels['qid'].isin(df['qid'])]
            for length, df in train_by_length.items()
        }

        print('Processing complete!')
        return {
            'corpus': corpus,  # collection of documents to be searched through - List
            'train_topics': train_topics,  # set of queries to use for training - Dataframe
            'val_topics': val_topics,  # set of queries for evaluation and tuning - Dataframe
            'test_topics': test_topics,  # queries for final evaluation - Dataframe
            'train_qrels': train_qrels,  # GROUND TRUTH - Dataframe
            'val_qrels': val_qrels,  # Query relevance judgments (qrels) that specify which documents are relevant to which queries. - Dataframe
            'test_qrels': test_qrels,  # Query relevance judgments (qrels) that specify which documents are relevant to which queries. - Dataframe
            'train_by_length': train_by_length,  # three training sets (queries) seperated by different lengths - Dataframe
            'train_qrels_by_length': train_qrels_by_length,  # three training sets (qrels) seperated by different lengths - Dataframe
            'index_variant': index_variant  # Indicates which index variant is being used for the dataset (e.g., "terrier_stemmed"). - String
        }

    def save_dataset_to_files(self, dataset: Dict[str, Any]):
        """
        Save various components of a dataset to files in organized directories.

        Args:
            dataset_info: Dictionary containing dataset components returned by load_dataset()
            dataset_name: Name of the dataset (e.g., 'msmarco', 'vaswani')
        """
        dataset_name = self.config.get('dataset')

        # Create necessary directories
        os.makedirs(f'../data/corpus/{dataset_name}', exist_ok=True)
        os.makedirs(f'../data/full_dataset/{dataset_name}', exist_ok=True)
        os.makedirs(f'../data/dataset_by_length/{dataset_name}', exist_ok=True)

        # Save corpus path
        with open(f'../data/corpus/{dataset_name}/corpus_path.txt', 'w') as f:
            f.write(str(dataset['corpus'][0]))

        # Merge the dataframes on the 'qid' column
        merged_train_df = pd.merge(dataset['train_topics'],dataset['train_qrels'],on='qid',how='inner').drop(columns=['query_length_category', 'label'], axis=1)
        merged_val_df = pd.merge(dataset['val_topics'],dataset['val_qrels'],on='qid',how='inner').drop(columns=['query_length_category', 'label'], axis=1)
        merged_test_df = pd.merge(dataset['test_topics'],dataset['test_qrels'],on='qid',how='inner').drop(columns=['query_length_category', 'label'], axis=1)

        # Save topics
        merged_train_df.to_csv(f'../data/full_dataset/{dataset_name}/train.csv', index=False)
        merged_val_df.to_csv(f'../data/full_dataset/{dataset_name}/val.csv', index=False)
        merged_test_df.to_csv(f'../data/full_dataset/{dataset_name}/test.csv', index=False)

        # Save train_by_length dataframes
        for topic, qrel in zip(dataset['train_by_length'].items(), dataset['train_qrels_by_length'].items()):
            os.makedirs(f'../data/dataset_by_length/{dataset_name}/{topic[0]}', exist_ok=True)
            # Merge the dataframes on the 'qid' column
            merged_df = pd.merge(topic[1],qrel[1],on='qid',how='inner').drop(columns=['query_length_category', 'label'], axis=1)
            merged_df.to_csv(f'../data/dataset_by_length/{dataset_name}/{topic[0]}/train.csv', index=False)


        # Save metadata about the dataset
        metadata = {
            'index_variant': dataset['index_variant'],
            'train_size': len(dataset['train_topics']),
            'val_size': len(dataset['val_topics']),
            'test_size': len(dataset['test_topics']),
            'short_queries': len(dataset['train_by_length'].get('short', pd.DataFrame())),
            'medium_queries': len(dataset['train_by_length'].get('medium', pd.DataFrame())),
            'long_queries': len(dataset['train_by_length'].get('long', pd.DataFrame()))
        }

        with open(f'../data/{dataset_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset {dataset_name} has been saved to disk.")
        print(f"Train topics: {metadata['train_size']}")
        print(f"Validation topics: {metadata['val_size']}")
        print(f"Test topics: {metadata['test_size']}")
        print(f"Query length distribution in training set:")
        print(f"  Short: {metadata['short_queries']} ({metadata['short_queries'] / metadata['train_size']:.1%})")
        print(f"  Medium: {metadata['medium_queries']} ({metadata['medium_queries'] / metadata['train_size']:.1%})")
        print(f"  Long: {metadata['long_queries']} ({metadata['long_queries'] / metadata['train_size']:.1%})")


