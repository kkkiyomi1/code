import numpy as np
import random
import logging

from data_utils import (
    load_mat_data,
    preprocess_data,
    multi_view_train_test_split,
    print_data_info
)
from fuzzy_system import TSKFuzzySystem
from representations import (
    extract_representations_random,
    extract_representations_pca,
    adaptive_weighted_combination
)
from clustering import (
    random_search_weights,
    find_best_k_using_silhouette,
    clustering_accuracy,
    calculate_purity
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    # ===== 1. Load data (example) =====
    file_path = "example_dataset.mat"  # Replace with your own .mat file
    views_data, labels = load_mat_data(file_path, data_key='X', label_key='Y')
    
    print_data_info(views_data, labels)
    
    # ===== 2. Preprocess data =====
    processed_views = preprocess_data(*views_data, method='standard', missing_value_strategy='mean')
    
    # (Optional) split train/test
    train_views, test_views, train_labels, test_labels = multi_view_train_test_split(
        processed_views, labels, test_size=0.3, random_state=seed
    )
    
    # ===== 3. Initialize TSK fuzzy system for each view =====

    n_clusters = 3
    membership_type = 'gaussian'  # or 'triangular'
    
    fuzzy_systems = []
    train_fuzzy_feats = []
    
    for vdata in train_views:
        fsys = TSKFuzzySystem(n_clusters=n_clusters, random_state=seed, membership_type=membership_type)
        fsys.fit(vdata)
        fz = fsys.transform(vdata)
        fuzzy_systems.append(fsys)
        train_fuzzy_feats.append(fz)
    
    # ===== 4. Extract representations =====

    shared_reps = []
    specific_reps = []
    
    for i, fz in enumerate(train_fuzzy_feats):
        if i < 2:
            shared, specific = extract_representations_random(fz, n_shared=20, n_specific=20)
        else:
            shared, specific = extract_representations_pca(fz, shared_dim=20, specific_dim=20)
        shared_reps.append(shared)
        specific_reps.append(specific)
    
    # ===== 5. (Optional) Find best k using silhouette =====

    data_for_k_selection = shared_reps[0].reshape(shared_reps[0].shape[0], -1)
    best_k, best_score = find_best_k_using_silhouette(data_for_k_selection, min_k=2, max_k=8, random_state=seed)
    logging.info(f"[Silhouette] best_k={best_k}, best_score={best_score:.4f}")
    
    # ===== 6. Random search for best weights =====

    best_w, best_nmi, best_acc, best_purity, best_ari = random_search_weights(
        shared_reps, specific_reps, train_labels,
        combiner_func=adaptive_weighted_combination,
        weight_range=(0.01, 0.99),
        n_iterations=50,
        n_clusters=best_k,
        random_state=seed,
        save_log=True,
        log_dir='search_logs'
    )
    
    logging.info("===== Random Search Results on Training Set =====")
    logging.info(f"Best Weights: {best_w}")
    logging.info(f"NMI={best_nmi:.4f}, ACC={best_acc:.4f}, PUR={best_purity:.4f}, ARI={best_ari:.4f}")
    
    # ===== 7. Evaluate on Test Set =====

    test_fuzzy_feats = []
    for i, vdata in enumerate(test_views):
        fz = fuzzy_systems[i].transform(vdata)
        test_fuzzy_feats.append(fz)
    

    test_shared_reps = []
    test_specific_reps = []
    
    for i, fz in enumerate(test_fuzzy_feats):
        if i < 2:
            shared_t, specific_t = extract_representations_random(fz, n_shared=20, n_specific=20)
        else:
            shared_t, specific_t = extract_representations_pca(fz, shared_dim=20, specific_dim=20)
        test_shared_reps.append(shared_t)
        test_specific_reps.append(specific_t)
    
    combined_test = adaptive_weighted_combination(test_shared_reps, test_specific_reps, best_w)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=seed)
    predicted_test_labels = kmeans.fit_predict(combined_test)
    
    nmi_test = np.round(random_search_weights.__globals__['normalized_mutual_info_score'](test_labels, predicted_test_labels), 4)
    acc_test = clustering_accuracy(test_labels, predicted_test_labels)
    pur_test = calculate_purity(test_labels, predicted_test_labels)
    
    logging.info("===== Test Set Evaluation =====")
    logging.info(f"NMI={nmi_test}, ACC={acc_test:.4f}, PUR={pur_test:.4f}")

if __name__ == "__main__":
    main()
