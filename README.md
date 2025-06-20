# DAGF-ABSA
Implementation of "DAGF: A Dual GCN and Auxiliary Graph Fusion Based Model for Aspect-Based Sentiment Analysis"



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==4.41.0
- einops==0.7.0
- cython==0.29.13
- nltk==3.5
- numpy==1.20.3
- spacy==3.7.4
- benepar==0.2.0

To install requirements, run `pip install -r requirements.txt`.

## Preparation

Prepare dataset with:

`python dataset/preprocess_data.py` and `python dataset/new_perocess_data.py`

The embedding.pt and stoi.pt files can be obtained from 

[APARN]: https://github.com/THU-BPM/APARN



## Training

To train the DAGFmodel, run:

`sh run.sh`
