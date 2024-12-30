import numpy as np
import scipy.io
import csv
import logging
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_mat_data(file_path, data_key='X', label_key='Y'):
    """
    Load multi-view data and labels from a .mat file.
    
    Parameters
    ----------
    file_path : str
        Path to the .mat file.
    data_key : str
        The key in the .mat file that contains multi-view data, assumed shape is (1, n_views).
    label_key : str
        The key in the .mat file that contains labels, assumed shape is (n_samples,) or (n_samples, 1).
    
    Returns
    -------
    views_data : list of np.ndarray
        A list of arrays, where each array corresponds to one view's data.
    labels : np.ndarray
        1D array of labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MAT file not found: {file_path}")
    
    logging.info(f"Loading .mat data from {file_path} ...")
    mat_data = scipy.io.loadmat(file_path)
    
    # Extract multi-view features
    # Assuming 'data_key' points to an object with shape (1, n_views)
    X = mat_data[data_key]
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError("The data_key in the .mat file does not match the assumed shape (1, n_views).")
    views_data = [X[0, i] for i in range(X.shape[1])]
    
    # Extract labels
    labels = mat_data[label_key].flatten()
    logging.info(f"Number of views: {len(views_data)} | Labels shape: {labels.shape}")
    
    return views_data, labels

def load_csv_data(file_path, delimiter=',', has_header=False):
    """
    Load data from a CSV file. This function can be adapted for multi-view data
    by splitting into different CSV files or columns.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str
        The character used to separate values. Default is ','.
    has_header : bool
        If True, the first row is treated as a header and skipped.
    
    Returns
    -------
    data : np.ndarray
        2D array containing the CSV data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logging.info(f"Loading CSV data from {file_path} ...")
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader)  # skip header
        for row in reader:
            float_row = [float(x) for x in row]
            rows.append(float_row)
    
    data = np.array(rows)
    logging.info(f"CSV data shape: {data.shape}")
    return data

def preprocess_data(*views, method='standard', missing_value_strategy='mean'):
    """
    Preprocess multiple views of data:
      1) Impute missing values
      2) Standardize or MinMax-scale the data
    
    Parameters
    ----------
    views : tuple of np.ndarray
        Each array is (n_samples, n_features) for one view.
    method : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    missing_value_strategy : str
        Strategy for imputing missing values, e.g. 'mean', 'median', 'most_frequent'
    
    Returns
    -------
    processed_views : list of np.ndarray
        The preprocessed views in the same order.
    """
    imputer = SimpleImputer(strategy=missing_value_strategy)
    processed_views = []
    
    for idx, view in enumerate(views):
        if not isinstance(view, np.ndarray):
            raise TypeError(f"View {idx+1} must be a NumPy array.")
        
        # Impute missing values
        view_imputed = imputer.fit_transform(view)
        
        # Standardize or MinMax scale
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        processed_view = scaler.fit_transform(view_imputed)
        processed_views.append(processed_view)
        logging.info(f"View {idx+1} preprocessed with {method} scaling.")
    
    return processed_views

def multi_view_train_test_split(views, labels, test_size=0.2, random_state=42):
    """
    Split multi-view data into train and test sets.
    Assumes all views have the same number of samples in the same order.
    
    Parameters
    ----------
    views : list of np.ndarray
        Each array is (n_samples, n_features).
    labels : np.ndarray
        1D array of labels.
    test_size : float
        Proportion of the data to be used as test set.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    views_train : list of np.ndarray
    views_test : list of np.ndarray
    labels_train : np.ndarray
    labels_test : np.ndarray
    """
    if len(views) == 0:
        raise ValueError("No views to split.")
    
    n_samples = views[0].shape[0]
    for i, v in enumerate(views):
        if v.shape[0] != n_samples:
            raise ValueError(f"View {i+1} does not match the sample size of the first view.")
    
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)

    views_train = [v[train_idx] for v in views]
    views_test = [v[test_idx] for v in views]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    
    logging.info(f"Data split into train (size={len(train_idx)}) and test (size={len(test_idx)})")
    return views_train, views_test, labels_train, labels_test

def print_data_info(views, labels):
    """
    Print basic information about multi-view data.
    
    Parameters
    ----------
    views : list of np.ndarray
        Each array is (n_samples, n_features).
    labels : np.ndarray
        1D array of labels.
    """
    logging.info("----- Data Info -----")
    for i, v in enumerate(views):
        logging.info(f"View {i+1}: shape={v.shape}, dtype={v.dtype}")
    logging.info(f"Labels: shape={labels.shape}, unique={np.unique(labels)}")
    logging.info("---------------------")
