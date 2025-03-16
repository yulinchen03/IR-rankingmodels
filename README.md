## To get started

1. Create a virtual environment:
```python
python -m venv IRvenv
```
2. Activate the virtual environment in the terminal:

Windows
```
IRvenv/Scripts/activate
```
macOS/Linux/Ubuntu
```python
source IRvenv/bin/activate
```

Then run the following:
```python
pip install -r requirements.txt
```

## To load and process datasets for training
Run the following
```python
python src/data_processor.py
```

### Optional: 
You can change the configurations in data_processor.py
in case you would like to work with another dataset from Pyterrier, 
set a different train/val/test split or test a different threshold for short/medium/long queries.