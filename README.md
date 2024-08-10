# UnitedDTA

# UnitedDTA: explainable multi-modal learning improves drug-target affinity predictions

# Requirements:
- python 3.8
- torch 2.2.0+cu118
- pytorch 1.10.0
- rdkit 2023.9.4
- networkx 3.1
- dgl 1.0.2+cu118
- deepchem 2.6.1
- mdanalysis 2.3.0
- scipy 1.9.1
- gensim 4.3.2

# How to run
## UnitedDTA
1. Run data_process.py to generate the preprocessed data.
2. Run label_process.py to generate train/test dataset and preprocessed label.
3. Run train_DTA_new.py to train and test the model.
