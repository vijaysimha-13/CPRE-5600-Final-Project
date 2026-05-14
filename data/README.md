# Dataset Folder

The full dataset files are not included in this repository because they are large.

This project uses two datasets:

## 1. Kaggle Network Intrusion Dataset

Dataset link:  
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download

Download the dataset from Kaggle and place the following files in this folder or in the project directory:

- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

These files are used for the main malicious vs. benign classification experiment.

## 2. CESNET-TLS22 Dataset

CESNET DataZoo link:  
https://cesnet.github.io/cesnet-datazoo/

The CESNET-TLS22 dataset is used for the TLS traffic metadata proof-of-concept experiment.

Install the required package:

```bash
pip install cesnet-datazoo
