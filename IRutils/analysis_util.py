# analysis_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from beir.datasets.data_loader import GenericDataLoader
from beir import util

class QueryLengthAnalyzer:
    def __init__(self, dataset_name, split="test", download_dir="datasets"):
        """
        Initialize the analyzer with a BEIR dataset.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.download_dir = download_dir
        self.docs, self.queries, self.qrels, self.df = self.load_dataset()
        self.compute_query_lengths()
        self.stats = self.compute_statistics()

    def load_dataset(self):
        """
        Download and load the BEIR dataset.
        Returns docs, queries, qrels, and a DataFrame of queries.
        """
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = util.download_and_unzip(url, self.download_dir)
        docs, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=self.split)
        df = pd.DataFrame(list(queries.items()), columns=["query_id", "query"])
        return docs, queries, qrels, df

    def compute_query_lengths(self):
        """
        Compute the query length (number of words) for each query.9
        Adds a new column 'query_length' to the DataFrame.
        """
        self.df['query_length'] = self.df['query'].apply(lambda x: len(str(x).split()))

    def compute_statistics(self):
        """
        Compute descriptive statistics for query lengths including 33rd and 67th percentiles.
        Returns a dictionary of statistics.
        """
        stats = self.df['query_length'].describe().to_dict()
        t1, t2 = np.percentile(self.df['query_length'], [33, 67])
        stats['33rd_percentile'] = t1
        stats['67th_percentile'] = t2
        return stats

    def plot_histogram(self, save_path=None):
        """
        Plot a histogram with KDE of query lengths.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['query_length'], bins=30, kde=True, color='skyblue')
        plt.title(f'{self.dataset_name.capitalize()} Query Length Distribution')
        plt.xlabel('Query Length (number of words)')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_boxplot(self, save_path=None):
        """
        Plot a box plot of query lengths.
        """
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df['query_length'], color='lightgreen')
        plt.title(f'{self.dataset_name.capitalize()} Query Length Box Plot')
        plt.xlabel('Query Length (number of words)')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def get_thresholds(self):
        """
        Returns thresholds for short, medium, and long queries based on 33rd and 67th percentiles.
        """
        t1 = self.stats['33rd_percentile']
        t2 = self.stats['67th_percentile']
        return {
            'short': (1, int(t1)),
            'medium': (int(t1) + 1, int(t2)),
            'long': (int(t2) + 1, self.df['query_length'].max())
        }

    def segment_queries(self):
        """
        Adds a column 'length_category' to the DataFrame based on computed thresholds.
        """
        thresholds = self.get_thresholds()
        def categorize(length):
            if length <= thresholds['short'][1]:
                return 'short'
            elif length <= thresholds['medium'][1]:
                return 'medium'
            else:
                return 'long'
        self.df['length_category'] = self.df['query_length'].apply(categorize)
        return self.df

# Example usage:
if __name__ == "__main__":
    analyzer = QueryLengthAnalyzer(dataset_name="quora")
    print("Statistics:", analyzer.stats)
    analyzer.plot_histogram()
    analyzer.plot_boxplot()
    df_segmented = analyzer.segment_queries()
    print("Category counts:\n", df_segmented['length_category'].value_counts())
