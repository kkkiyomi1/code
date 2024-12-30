# Multi-View Unsupervised Representation Learning via the Integration of Fuzzy Rules and Graph-Based Adaptive Regularization

> **A Python library for multi-view fuzzy clustering, featuring TSK-based fuzzy systems, shared/specific representation extraction, and advanced cluster evaluation.**

This repository provides an end-to-end pipeline for fuzzy clustering on multi-view data. It is designed to serve as a highly extensible and easy-to-read foundation for academic papers or research projects. The project adopts a modular structure, dividing functionalities such as data loading/preprocessing, TSK fuzzy systems, shared/specific representation extraction, clustering, and evaluation into separate files. A comprehensive example of how to run the entire process is demonstrated in `demo.py`.

---

## Features

- **Multi-View Support**: Simultaneously handles multiple views of data (e.g., image features, text features, etc.).  
- **TSK Fuzzy System**: Initializes membership centers via KMeans, supports Gaussian membership functions, and can be extended to other fuzzy membership functions.  
- **Shared & Specific Representation Extraction**: Uses random projection, PCA, or can be extended to other methods (e.g., AutoEncoder).  
- **Rich Clustering Evaluation**: NMI, ACC, Purity, ARI, Silhouette Score, plus random or grid search to optimize view weights.  
- **Modular Design**: Split into multiple files for clarity, maintainability, and easier further development.  
- **Detailed Logging & Documentation**: Extensive code comments and `logging` outputs help with debugging, tracking, and understanding the process.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Descriptions](#module-descriptions)
- [Citation](#citation)
- [License](#license)

---

## Project Structure

```text
multi_view_fuzzy_clustering/
├── data_utils.py            # Data loading & preprocessing functions
├── fuzzy_system.py          # TSK fuzzy system logic (classes + functions)
├── representations.py       # Shared & specific representation extraction
├── clustering.py            # Clustering, metrics, random/grid search, etc.
└── demo.py                  # Usage example script
