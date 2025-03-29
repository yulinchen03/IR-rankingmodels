import random

class DataProcessor:
    def __init__(self, queries, documents, qrels):
        self.queries = queries
        self.documents = documents
        self.qrels = qrels

    def get_testset(self, test_ratio=0.2, random_state=42):
        keys = list(self.queries.keys())
        num_samples = len(keys)

        if random_state is not None:
            random.seed(random_state)
        random.shuffle(keys)

        test_size = int(num_samples * test_ratio)
        test_keys = keys[:test_size]
        query_test = {k: self.queries[k] for k in test_keys}
        qrel_test = {k: self.qrels[k] for k in test_keys}

        remaining_keys = list(self.queries.keys())[test_size:]
        self.queries = {k: self.queries[k] for k in remaining_keys}
        self.qrels = {k: self.qrels[k] for k in remaining_keys}

        return query_test, qrel_test


    def get_subset(self, min_len, max_len, full=False):

        filtered_queries = {qid: q for qid, q in self.queries.items() if min_len < len(q.split()) <= max_len}

        # Get the filtered query IDs based on length
        filtered_query_ids = [qid for qid, q in self.queries.items() if min_len < len(q.split()) <= max_len]

        # Filter qrels based on the filtered query IDs
        filtered_qrels = {qid: self.qrels[qid] for qid in filtered_query_ids if qid in self.qrels}

        self.queries, self.qrels = filtered_queries, filtered_qrels

        return filtered_queries, filtered_qrels

    def train_val_split(self, train_ratio=0.8, val_ratio=0.2, random_state=None):
        if abs(train_ratio + val_ratio - 1.0) > 1e-9:
            raise ValueError("Train, validation ratios must sum to 1.")

        keys = list(self.queries.keys())
        num_samples = len(keys)

        if random_state is not None:
            random.seed(random_state)
        random.shuffle(keys)

        # if dataset is used solely for training (only when seperate test set available)
        train_size = int(num_samples * train_ratio)

        train_keys = keys[:train_size]
        val_keys = keys[train_size:]

        query_train = {k: self.queries[k] for k in train_keys}
        query_val = {k: self.queries[k] for k in val_keys}

        qrel_train = {k: self.qrels[k] for k in train_keys}
        qrel_val = {k: self.qrels[k] for k in val_keys}


        return query_train, query_val, qrel_train, qrel_val