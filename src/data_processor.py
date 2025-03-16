from utils.create_datasets import QueryLengthClassifier, DataManager
import os

def main(config):
    dm = DataManager(config, QueryLengthClassifier(config))
    dataset = dm.load_dataset()
    dm.save_dataset_to_files(dataset)

# run this script to get a dataset and create train-val-test splits as well as split by length datasets
if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-19"
    config = {
        "short_max": 4, # adjust if needed
        "medium_max": 7, # adjust if needed
        "dataset": "msmarco", # adjust if needed
        "dataset_variant": "terrier_stemmed",
        "topic_variant": "dev.small",  # change once the pipeline is complete to access large dataset (currently ~7000 samples)
        "test_size": 0.2,  # adjust if needed
        "val_size": 0.2  # adjust if needed
    }
    main(config)