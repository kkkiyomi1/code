import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def gaussian_membership(x, center, width):
    """
    Gaussian membership function.
    
    Parameters
    ----------
    x : np.ndarray
        Shape (n_features,)
    center : np.ndarray
        Shape (n_features,)
    width : float or np.ndarray
        Standard deviation of the Gaussian function
    
    Returns
    -------
    membership : np.ndarray
        Element-wise membership score, same shape as x
    """
    return np.exp(-np.square(x - center) / (2 * np.square(width)))

def trimf_membership(x, a, b, c):
    """
    Triangular membership function.
    The membership is 0 outside [a, c], with a peak at b.
    
    Parameters
    ----------
    x : np.ndarray
        Shape (n_features,)
    a : float
        Left foot of the triangle
    b : float
        Peak
    c : float
        Right foot of the triangle
    
    Returns
    -------
    membership : np.ndarray
        Element-wise membership score, shape same as x
    """
    # To make it more advanced, handle vector input
    mem = np.zeros_like(x)
    # left slope
    left_mask = (x >= a) & (x <= b)
    mem[left_mask] = (x[left_mask] - a) / (b - a + 1e-9)
    # right slope
    right_mask = (x >= b) & (x <= c)
    mem[right_mask] = (c - x[right_mask]) / (c - b + 1e-9)
    mem[x < a] = 0
    mem[x > c] = 0
    return mem

class TSKFuzzySystem:
    """
    A TSK-based fuzzy system that initializes fuzzy centers using KMeans
    and transforms data into a fuzzy feature space. Can handle multiple membership
    function types: 'gaussian' or 'triangular'.
    
    Attributes
    ----------
    n_clusters : int
        Number of clusters (rules) in TSK fuzzy system
    random_state : int
        Random seed
    membership_type : str
        Either 'gaussian' or 'triangular' membership
    centers_ : np.ndarray
        Shape (n_clusters, n_features), cluster centers
    widths_ : np.ndarray
        Shape (n_clusters,), or could be per-dimension if extended
    params_ : list of tuples
        For triangular membership, each cluster might have (a, b, c)
    is_initialized_ : bool
        Whether the system has been fitted
    """
    def __init__(self, n_clusters=3, random_state=42, membership_type='gaussian'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.membership_type = membership_type
        
        self.centers_ = None
        self.widths_ = None
        self.params_ = None
        self.is_initialized_ = False

    def fit(self, data):
        """
        Fit the fuzzy system using KMeans to find centers and compute widths or triangular parameters.
        
        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features)
        """
        logging.info(f"Fitting TSKFuzzySystem with {self.n_clusters} clusters using {self.membership_type} membership.")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(data)
        centers = kmeans.cluster_centers_
        
        if self.membership_type == 'gaussian':
            # Calculate average distance to set widths
            distances = pairwise_distances(centers, centers)
            np.fill_diagonal(distances, np.inf)
            valid_distances = distances[~np.isinf(distances)]
            if valid_distances.size == 0:
                raise ValueError("No valid distances found between cluster centers.")
            widths = np.mean(valid_distances) * np.ones(self.n_clusters)
            
            self.centers_ = centers
            self.widths_ = widths
            self.params_ = None
            self.is_initialized_ = True
            logging.info("Gaussian-based TSKFuzzySystem successfully initialized.")
        
        elif self.membership_type == 'triangular':

            distances = pairwise_distances(centers, centers)
            np.fill_diagonal(distances, np.inf)
            valid_distances = distances[~np.isinf(distances)]
            if valid_distances.size == 0:
                raise ValueError("No valid distances found between cluster centers.")
            d = np.mean(valid_distances) / 2.0
            


            cluster_params = []
            for center in centers:

                avg_center = np.mean(center)
                cluster_params.append((avg_center - d, avg_center, avg_center + d))
            
            self.centers_ = centers
            self.widths_ = None
            self.params_ = cluster_params
            self.is_initialized_ = True
            logging.info("Triangular-based TSKFuzzySystem successfully initialized.")
        else:
            raise ValueError(f"Unknown membership_type: {self.membership_type}")

    def transform(self, data):
        """
        Transform data into fuzzy feature space.
        
        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features)
        
        Returns
        -------
        fuzzy_features : np.ndarray
            Shape (n_samples, n_clusters, n_features), 
            where each sample has membership vectors for each cluster.
        """
        if not self.is_initialized_:
            raise RuntimeError("TSKFuzzySystem not fitted. Call `fit` before `transform`.")
        
        logging.info("Transforming data into fuzzy space...")
        fuzzy_list = []
        for x in data:
            # membership_list will be shape (n_clusters, n_features)
            membership_list = []
            if self.membership_type == 'gaussian':
                for i in range(self.n_clusters):
                    membership_list.append(
                        gaussian_membership(x, self.centers_[i], self.widths_[i])
                    )
            elif self.membership_type == 'triangular':
                for i in range(self.n_clusters):
                    (a, b, c) = self.params_[i]
                    membership_list.append(
                        trimf_membership(x, a, b, c)
                    )
            membership_list = np.array(membership_list)  # shape (n_clusters, n_features)
            fuzzy_list.append(membership_list)
        
        fuzzy_features = np.array(fuzzy_list)  # shape (n_samples, n_clusters, n_features)
        return fuzzy_features

    def fit_transform(self, data):
        """
        Combine fit() and transform() in one step.
        """
        self.fit(data)
        return self.transform(data)
