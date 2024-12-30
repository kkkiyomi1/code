import numpy as np
import random
import logging
import json
import os

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, accuracy_score, silhouette_score, adjusted_rand_score
)
from scipy.stats import mode

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def clustering_accuracy(true_labels, predicted_labels):
    """
    Calculate clustering accuracy using majority vote within each cluster.
    
    Parameters
    ----------
    true_labels : np.ndarray
        Ground-truth labels, shape (n_samples,)
    predicted_labels : np.ndarray
        Cluster assignments, shape (n_samples,)
    
    Returns
    -------
    acc : float
        Clustering accuracy
    """
    match_labels = np.zeros_like(predicted_labels)
    for clus_id in np.unique(predicted_labels):
        mask = (predicted_labels == clus_id)
        # mode returns (mode_val, count)
        majority_label = mode(true_labels[mask], keepdims=True).mode[0]
        match_labels[mask] = majority_label
    return accuracy_score(true_labels, match_labels)

def calculate_purity(true_labels, predicted_labels):
    """
    Calculate clustering purity.
    
    Parameters
    ----------
    true_labels : np.ndarray
    predicted_labels : np.ndarray
    
    Returns
    -------
    purity : float
    """
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)
    
    # Build contingency matrix
    contingency = np.zeros((len(unique_true), len(unique_pred)))
    
    # Map labels to indices
    true_to_idx = {lbl: i for i, lbl in enumerate(unique_true)}
    pred_to_idx = {lbl: i for i, lbl in enumerate(unique_pred)}
    
    for t, p in zip(true_labels, predicted_labels):
        contingency[true_to_idx[t], pred_to_idx[p]] += 1
    
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def random_search_weights(shared_reps, specific_reps, true_labels, combiner_func,
                          weight_range=(0.01, 0.99), n_iterations=100,
                          n_clusters=10, random_state=42,
                          save_log=False, log_dir='logs'):
    """
    Randomly search for the optimal view weights to maximize clustering metrics (NMI, ACC, Purity, ARI).
    
    Parameters
    ----------
    shared_reps : list of np.ndarray
        Shared representations for each view.
    specific_reps : list of np.ndarray
        Specific representations for each view.
    true_labels : np.ndarray
        Ground-truth labels.
    combiner_func : function
        A function that takes (shared_reps, specific_reps, weights) and returns a combined representation.
    weight_range : tuple
        Min and max for random uniform sampling of weights.
    n_iterations : int
        Number of random search iterations.
    n_clusters : int
        Number of clusters for KMeans.
    random_state : int
        Random seed.
    save_log : bool
        If True, saves intermediate results to a JSON file.
    log_dir : str
        Directory for logs.
    
    Returns
    -------
    best_weights : np.ndarray
    best_nmi : float
    best_acc : float
    best_purity : float
    best_ari : float
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    if save_log and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    best_nmi, best_acc, best_purity, best_ari = -1, -1, -1, -1
    best_weights = None
    
    for i in range(n_iterations):
        # Random generate weights
        w = np.array([random.uniform(*weight_range) for _ in range(len(shared_reps))])
        # Normalize
        w = w / w.sum()
        
        combined = combiner_func(shared_reps, specific_reps, w)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        predicted_labels = kmeans.fit_predict(combined)
        
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        acc = clustering_accuracy(true_labels, predicted_labels)
        pur = calculate_purity(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        
        logging.info(f"[Iter {i+1}/{n_iterations}] Weights={w}, NMI={nmi:.4f}, ACC={acc:.4f}, PUR={pur:.4f}, ARI={ari:.4f}")
        
        # Update best
        if nmi > best_nmi:
            best_nmi = nmi
            best_acc = acc
            best_purity = pur
            best_ari = ari
            best_weights = w
        
        if save_log:
            record = {
                'iteration': i+1,
                'weights': w.tolist(),
                'NMI': nmi,
                'ACC': acc,
                'PUR': pur,
                'ARI': ari
            }
            with open(os.path.join(log_dir, 'random_search_log.json'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
    
    return best_weights, best_nmi, best_acc, best_purity, best_ari

def grid_search_weights(shared_reps, specific_reps, true_labels, combiner_func,
                        weight_candidates, n_clusters=10, random_state=42):
    """
    Performs a grid search for multi-view weights. Each candidate in weight_candidates is a list or array
    that sums to 1 (or it will be normalized internally).
    
    Parameters
    ----------
    shared_reps : list of np.ndarray
    specific_reps : list of np.ndarray
    true_labels : np.ndarray
    combiner_func : function
    weight_candidates : list of list
        For instance: [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], ...]
    n_clusters : int
    random_state : int
    
    Returns
    -------
    best_weights : np.ndarray
    best_nmi : float
    """
    logging.info("Starting grid search for weights.")
    best_nmi = -1
    best_weights = None
    
    for idx, candidate in enumerate(weight_candidates):
        candidate = np.array(candidate)
        candidate = candidate / candidate.sum()  # normalize
        combined = combiner_func(shared_reps, specific_reps, candidate)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(combined)
        
        nmi = normalized_mutual_info_score(true_labels, labels)
        logging.info(f"[Candidate {idx+1}] Weights={candidate}, NMI={nmi:.4f}")
        
        if nmi > best_nmi:
            best_nmi = nmi
            best_weights = candidate
    
    return best_weights, best_nmi

def find_best_k_using_silhouette(data, min_k=2, max_k=20, random_state=42):
    """
    Find the optimal number of clusters k by evaluating Silhouette Score.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_features)
    min_k : int
    max_k : int
    random_state : int
    
    Returns
    -------
    best_k : int
    best_score : float
    """
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        logging.info(f"k={k}, Silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score
