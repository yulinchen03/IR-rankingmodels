# Information Retrieval Experiment
April 2025, TU Delft

This repository contains scripts and data for our experiments as described in our paper:

[`Beyond Monolithic Finetuning: Evaluating Query-Length-Specific Finetuning and Ensembling for Neural Retrieval Models`](report.pdf)


## To get started

1. Create a virtual environment:
```python
python -m venv .venv
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

## Train/Finetune models
Refer to [train.py](train.py) for details.

To start training right away:
```bash
python train.py
```

## Run experiments
Refer to [ensemble.py](ensemble.py) for details

To start running the experiments right away:
```bash
python ensemble.py
```

## Note
Please make sure you have cuda enabled: https://docs.nvidia.com/cuda/cuda-quick-start-guide/

Our code and parameter configurations were tested with the following hardware:

- Processor: AMD Ryzen Threadripper PRO 3975WX
- Memory: 128GB with 20GB extra swap
- GPU: NVIDIA Quadro RTX A6000 48GB

Disclaimer: The experiments presented are resource demanding. Please make sure your PC can support the workload before you begin running the scripts as it has resulted in multiple system crashes in the past on our machines.

## Contributors
***Information Retrieval Group 19***
- Serkan Akin
- Elvira Antonogiannaki
- Yiming Chen
- Yulin Chen

