import numpy as np
import logging

from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    # If umap is not installed, we handle it gracefully.
    UMAP_AVAILABLE = False

try:
    # t-SNE is in sklearn.manifold
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def extract_representations_random(fuzzy_features, n_shared=20, n_specific=20, use_sklearn=False):
    """
    Extract shared and specific representations via random projection (RP).
    
    This function supports two approaches:
    1) A custom random projection using a manual random matrix.
    2) (Optional) Use scikit-learn's GaussianRandomProjection as an alternative.

    Parameters
    ----------
    fuzzy_features : np.ndarray
        Shape (n_samples, n_rules, n_feats)
    n_shared : int
        Dimensionality of the 'shared' representation
    n_specific : int
        Dimensionality of the 'specific' representation
    use_sklearn : bool
        If True, uses GaussianRandomProjection from sklearn for the 'shared' part
        and a custom random matrix for 'specific' part, for the shared part.
        If False, both parts use custom random matrices.

    Returns
    -------
    shared_representation : np.ndarray
        Shape (n_samples, n_rules, n_shared) or (n_samples, n_shared)
    specific_representation : np.ndarray
        Shape (n_samples, n_rules, n_specific) or (n_samples, n_specific)
    """
    logging.info("Extracting representations via Random Projection.")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    
    # Flatten each sample from (n_rules, n_feats) => (n_rules*n_feats,)
    flattened = fuzzy_features.reshape(n_samples, n_rules*n_feats)

    # Shared representation
    if use_sklearn:
        # Use scikit-learn's GaussianRandomProjection
        logging.info(f"Using GaussianRandomProjection for shared representation (n_components={n_shared}).")
        grp = GaussianRandomProjection(n_components=n_shared)
        shared_rep = grp.fit_transform(flattened)  # shape (n_samples, n_shared)
    else:
        # Custom random matrix
        logging.info(f"Using custom random matrix for shared representation (dim={n_shared}).")
        proj_shared = np.random.randn(n_rules*n_feats, n_shared)
        shared_rep = flattened @ proj_shared  # shape (n_samples, n_shared)

    # Specific representation: also random, but we keep it separate
    logging.info(f"Using custom random matrix for specific representation (dim={n_specific}).")
    proj_specific = np.random.randn(n_rules*n_feats, n_specific)
    specific_rep = flattened @ proj_specific  # shape (n_samples, n_specific)

    return shared_rep, specific_rep

def extract_representations_pca(fuzzy_features, shared_dim=20, specific_dim=20):

    logging.info("Extracting representations using PCA.")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    
    # Flatten
    flattened = fuzzy_features.reshape(n_samples, -1)

    # PCA for shared
    pca_shared = PCA(n_components=shared_dim)
    shared_rep = pca_shared.fit_transform(flattened)
    
    # PCA for specific
    pca_specific = PCA(n_components=specific_dim)
    specific_rep = pca_specific.fit_transform(flattened)
    
    return shared_rep, specific_rep

def extract_representations_kernel_pca(fuzzy_features, shared_dim=20, specific_dim=20, kernel='rbf'):
    """
    Extract shared and specific representations using Kernel PCA with a chosen kernel.

    Parameters
    ----------
    fuzzy_features : np.ndarray
        (n_samples, n_rules, n_feats)
    shared_dim : int
    specific_dim : int
    kernel : str
        Kernel type for KernelPCA ('linear', 'poly', 'rbf', 'sigmoid', etc.)

    Returns
    -------
    shared_rep : np.ndarray
        (n_samples, shared_dim)
    specific_rep : np.ndarray
        (n_samples, specific_dim)
    """
    logging.info(f"Extracting representations using Kernel PCA with kernel='{kernel}'.")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    flattened = fuzzy_features.reshape(n_samples, -1)

    kpca_shared = KernelPCA(n_components=shared_dim, kernel=kernel)
    shared_rep = kpca_shared.fit_transform(flattened)

    kpca_specific = KernelPCA(n_components=specific_dim, kernel=kernel)
    specific_rep = kpca_specific.fit_transform(flattened)

    return shared_rep, specific_rep

def extract_representations_ica(fuzzy_features, shared_dim=20, specific_dim=20):
    """
    Extract shared and specific representations using FastICA (Independent Component Analysis).
    Demonstrates how to use different transformations for each part.

    Parameters
    ----------
    fuzzy_features : np.ndarray
        (n_samples, n_rules, n_feats)
    shared_dim : int
    specific_dim : int

    Returns
    -------
    shared_rep : np.ndarray
        (n_samples, shared_dim)
    specific_rep : np.ndarray
        (n_samples, specific_dim)
    """
    logging.info("Extracting representations using FastICA.")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    flattened = fuzzy_features.reshape(n_samples, -1)

    ica_shared = FastICA(n_components=shared_dim, random_state=42)
    shared_rep = ica_shared.fit_transform(flattened)

    ica_specific = FastICA(n_components=specific_dim, random_state=42)
    specific_rep = ica_specific.fit_transform(flattened)

    return shared_rep, specific_rep

def extract_representations_umap(fuzzy_features, shared_dim=20, specific_dim=20, n_neighbors=15, min_dist=0.1):
    """
    Extract shared and specific representations using UMAP (Uniform Manifold Approximation and Projection).
    This requires 'umap-learn' to be installed.

    Parameters
    ----------
    fuzzy_features : np.ndarray
        (n_samples, n_rules, n_feats)
    shared_dim : int
    specific_dim : int
    n_neighbors : int
        UMAP parameter: number of neighbors
    min_dist : float
        UMAP parameter: minimum distance between points in the low-dimensional space

    Returns
    -------
    shared_rep : np.ndarray
        (n_samples, shared_dim)
    specific_rep : np.ndarray
        (n_samples, specific_dim)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Please install via 'pip install umap-learn' to use this method.")
    
    logging.info("Extracting representations using UMAP.")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    flattened = fuzzy_features.reshape(n_samples, -1)

    # UMAP for shared
    umap_shared = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=shared_dim, random_state=42)
    shared_rep = umap_shared.fit_transform(flattened)

    # UMAP for specific
    umap_specific = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=specific_dim, random_state=42)
    specific_rep = umap_specific.fit_transform(flattened)

    return shared_rep, specific_rep

def extract_representations_tsne(fuzzy_features, shared_dim=20, specific_dim=20, perplexity=30.0):
    """
    Extract shared and specific representations using t-SNE.
    Typically used for visualization in 2D or 3D, but we allow arbitrary dimensions here.

    Note: t-SNE can be slow and is not guaranteed to preserve global structure.

    Parameters
    ----------
    fuzzy_features : np.ndarray
        (n_samples, n_rules, n_feats)
    shared_dim : int
    specific_dim : int
    perplexity : float
        t-SNE perplexity parameter

    Returns
    -------
    shared_rep : np.ndarray
        (n_samples, shared_dim)
    specific_rep : np.ndarray
        (n_samples, specific_dim)
    """
    if not TSNE_AVAILABLE:
        raise ImportError("t-SNE is not available. Please check your scikit-learn version or install the appropriate package.")
    
    logging.info("Extracting representations using t-SNE. (Warning: this can be slow for large n_samples.)")
    n_samples, n_rules, n_feats = fuzzy_features.shape
    flattened = fuzzy_features.reshape(n_samples, -1)

    tsne_shared = TSNE(n_components=shared_dim, perplexity=perplexity, random_state=42)
    shared_rep = tsne_shared.fit_transform(flattened)

    tsne_specific = TSNE(n_components=specific_dim, perplexity=perplexity, random_state=42)
    specific_rep = tsne_specific.fit_transform(flattened)

    return shared_rep, specific_rep

def adaptive_weighted_combination(shared_reps, specific_reps, weights):
    """
    Perform a weighted combination of multi-view shared/specific representations.
    
    """
    if len(shared_reps) != len(specific_reps):
        raise ValueError("shared_reps and specific_reps must have the same length.")
    if len(weights) != len(shared_reps):
        raise ValueError("weights length must match the number of views.")
    
    logging.info("Performing adaptive weighted combination of multi-view representations.")
    combined_list = []
    for i in range(len(shared_reps)):
        w = weights[i]
        shared = shared_reps[i]
        specific = specific_reps[i]
        
        # Flatten or keep shape depending on your design
        shared_flat = shared.reshape(shared.shape[0], -1)
        specific_flat = specific.reshape(specific.shape[0], -1)

        combined_view = w * (shared_flat + specific_flat)
        combined_list.append(combined_view)

    # Sum up all views
    final_combined_representation = sum(combined_list)
    return final_combined_representation

class AdvancedRepresentationExtractor:
    """
    A class for advanced representation extraction, supporting multiple algorithms:
      - 'random' (internal random projection or sklearn GaussianRandomProjection)
      - 'pca'
      - 'kernel_pca'
      - 'ica'
      - 'umap'
      - 'tsne'
    
    This class can 'fit' on a training fuzzy_features set and then 'transform' 
    new data in a consistent manner (where applicable). For methods that do not 
    produce a consistent transformer (like t-SNE in some versions), we re-fit 
    or raise an error.
    """
    def __init__(self, mode='random', shared_dim=20, specific_dim=20, **kwargs):
        """
        Parameters
        ----------
        mode : str
            One of {'random', 'pca', 'kernel_pca', 'ica', 'umap', 'tsne'}
        shared_dim : int
        specific_dim : int
        kwargs : dict
            Additional parameters for the respective method (e.g., kernel='rbf', perplexity=30.0, etc.)
        """
        self.mode = mode
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.kwargs = kwargs

        self._fitted_shared_model = None
        self._fitted_specific_model = None
        self._already_fitted = False

    def fit(self, fuzzy_features):
        """
        Fit internal models (if the chosen mode supports a 'fit' + 'transform' lifecycle).
        For t-SNE or UMAP, we typically 'fit_transform' at once. Some can store embeddings, 
        but re-transforming new samples might not be straightforward. 
        This is a simplified approach.

        Parameters
        ----------
        fuzzy_features : np.ndarray
            (n_samples, n_rules, n_feats)
        """
        logging.info(f"Fitting AdvancedRepresentationExtractor with mode='{self.mode}'.")
        n_samples, n_rules, n_feats = fuzzy_features.shape
        flattened = fuzzy_features.reshape(n_samples, -1)

        if self.mode == 'random':
            # Typically random approach does not need 'fit' but we store random matrices
            use_sklearn = self.kwargs.get('use_sklearn', False)
            if use_sklearn:
                # For demonstration, we store the scikit GRP models
                self._fitted_shared_model = GaussianRandomProjection(
                    n_components=self.shared_dim,
                    random_state=42
                ).fit(flattened)
            else:
                # Store random matrices
                self._fitted_shared_model = np.random.randn(flattened.shape[1], self.shared_dim)
            
            # For specific part, always store custom random matrix
            self._fitted_specific_model = np.random.randn(flattened.shape[1], self.specific_dim)
            
            self._already_fitted = True

        elif self.mode == 'pca':
            # We keep separate PCA models for shared and specific
            pca_shared = PCA(n_components=self.shared_dim)
            pca_shared.fit(flattened)
            pca_specific = PCA(n_components=self.specific_dim)
            pca_specific.fit(flattened)
            self._fitted_shared_model = pca_shared
            self._fitted_specific_model = pca_specific
            self._already_fitted = True

        elif self.mode == 'kernel_pca':
            kernel = self.kwargs.get('kernel', 'rbf')
            kpca_shared = KernelPCA(n_components=self.shared_dim, kernel=kernel)
            kpca_shared.fit(flattened)
            kpca_specific = KernelPCA(n_components=self.specific_dim, kernel=kernel)
            kpca_specific.fit(flattened)
            self._fitted_shared_model = kpca_shared
            self._fitted_specific_model = kpca_specific
            self._already_fitted = True

        elif self.mode == 'ica':
            ica_shared = FastICA(n_components=self.shared_dim, random_state=42)
            ica_shared.fit(flattened)
            ica_specific = FastICA(n_components=self.specific_dim, random_state=42)
            ica_specific.fit(flattened)
            self._fitted_shared_model = ica_shared
            self._fitted_specific_model = ica_specific
            self._already_fitted = True

        elif self.mode == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Please install 'umap-learn'.")
            # For UMAP, we typically do fit_transform once. 
            # We store it, but re-transforming new data might not always be consistent.
            # Some newer versions of UMAP do provide partial transform.
            self._fitted_shared_model = umap.UMAP(
                n_neighbors=self.kwargs.get('n_neighbors', 15),
                min_dist=self.kwargs.get('min_dist', 0.1),
                n_components=self.shared_dim,
                random_state=42
            )
            self._fitted_shared_model.fit(flattened)

            self._fitted_specific_model = umap.UMAP(
                n_neighbors=self.kwargs.get('n_neighbors', 15),
                min_dist=self.kwargs.get('min_dist', 0.1),
                n_components=self.specific_dim,
                random_state=42
            )
            self._fitted_specific_model.fit(flattened)
            self._already_fitted = True
        
        elif self.mode == 'tsne':
            if not TSNE_AVAILABLE:
                raise ImportError("t-SNE is not installed or not available in sklearn.manifold.")
            # Similar approach to UMAP: t-SNE doesn't strictly provide 'transform' in older versions.
            # We'll just store the placeholders to indicate we've 'fit' once.
            self._fitted_shared_model = TSNE(
                n_components=self.shared_dim,
                perplexity=self.kwargs.get('perplexity', 30.0),
                random_state=42
            )
            self._fitted_specific_model = TSNE(
                n_components=self.specific_dim,
                perplexity=self.kwargs.get('perplexity', 30.0),
                random_state=42
            )
            # We do not call fit here because t-SNE is typically done via fit_transform
            # We'll do that in transform() for consistency.
            self._already_fitted = True
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def transform(self, fuzzy_features):
        """
        Transform fuzzy_features into shared and specific representations 
        using the fitted model(s).

        Parameters
        ----------
        fuzzy_features : np.ndarray
            (n_samples, n_rules, n_feats)

        Returns
        -------
        shared_rep : np.ndarray
            (n_samples, shared_dim)
        specific_rep : np.ndarray
            (n_samples, specific_dim)
        """
        if not self._already_fitted:
            raise RuntimeError("You must call fit() before transform() for this extractor.")
        
        n_samples, n_rules, n_feats = fuzzy_features.shape
        flattened = fuzzy_features.reshape(n_samples, -1)

        if self.mode == 'random':
            # Check what we stored in _fitted_shared_model
            if isinstance(self._fitted_shared_model, np.ndarray):
                # custom random matrix
                shared_rep = flattened @ self._fitted_shared_model
            else:
                # sklearn's GaussianRandomProjection
                shared_rep = self._fitted_shared_model.transform(flattened)

            specific_rep = flattened @ self._fitted_specific_model

        elif self.mode in ['pca', 'kernel_pca', 'ica']:
            shared_rep = self._fitted_shared_model.transform(flattened)
            specific_rep = self._fitted_specific_model.transform(flattened)

        elif self.mode == 'umap':
            # Some versions of UMAP can do .transform(), some do not. 
            # Here we assume the version that supports transform.
            shared_rep = self._fitted_shared_model.transform(flattened)
            specific_rep = self._fitted_specific_model.transform(flattened)

        elif self.mode == 'tsne':
            # Typically t-SNE doesn't have a 'transform()' separate from 'fit_transform()'.
            # We'll do 'fit_transform()' each time or raise a warning.
            logging.warning("t-SNE does not traditionally support separate transform(). We'll do fit_transform again.")
            shared_rep = self._fitted_shared_model.fit_transform(flattened)
            specific_rep = self._fitted_specific_model.fit_transform(flattened)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return shared_rep, specific_rep

    def fit_transform(self, fuzzy_features):
        """
        Convenience method to do both fit and transform in one go.
        """
        self.fit(fuzzy_features)
        return self.transform(fuzzy_features)
