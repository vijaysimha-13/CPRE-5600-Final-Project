# TLS Fingerprinting for Malicious vs. Benign Traffic Classification

## Project Overview

This project investigates metadata-based machine learning for encrypted network traffic analysis. The main objective is to classify network traffic as benign or malicious using observable traffic metadata without decrypting packet payloads.

The project includes two related experiments:

1. **Kaggle Network Intrusion Dataset Experiment**  
   This is the main malicious vs. benign classification experiment. It uses CICIDS/ISCX-style flow-level traffic records from the Kaggle Network Intrusion Dataset. The original dataset labels were converted into binary classes:
   - `BENIGN` = 0
   - All attack labels = 1

2. **CESNET-TLS22 TLS Traffic Metadata Proof-of-Concept**  
   This experiment was added to better align the project with TLS fingerprinting. CESNET-TLS22 provides real TLS traffic metadata and application/service labels. Since the loaded CESNET-TLS22 subset does not provide verified malicious and benign labels, this experiment is reported as a TLS traffic metadata proof-of-concept rather than direct malicious traffic detection.

The machine learning models used in this project are:

- Decision Tree
- Random Forest
- XGBoost

The project also includes a **synthetic perturbation stress test**, where random noise was added to test features to evaluate how sensitive the trained model is when the feature distribution changes. This is not treated as true real-world concept drift.

---

## Repository Contents

```text
TLS-Fingerprinting-Malicious-Traffic-Detection/
│
├── README.md
├── Final_Project_Report.pdf
├── TLS_Fingerprinting_Modeling.ipynb
├── tls_fingerprinting_modeling.py
├── requirements.txt
│
├── outputs/
│   ├── confusion_matrix_cicids.png
│   ├── roc_curve_cicids.png
│   ├── feature_importance_cicids.png
│   ├── confusion_matrix_perturbation_cicids.png
│   ├── confusion_matrix_cesnet_tls.png
│   ├── roc_curve_cesnet_tls.png
│   └── feature_importance_cesnet_tls.png
│
└── data/
    └── README.md
```

Datasets used and links to access them are included in the data/README.md file.