import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet


def correlation_to_distance(corr_matrix):
    """
    Maps a Pearson correlation matrix to a distance metric space.
    Using the formula: d(i,j) = sqrt(2 * (1- rho_ij))
    """

    #Extract raw numpy array for faster computations
    raw_corr = corr_matrix.values

    #clip values to avoid numerical issues
    safe_corr = np.clip(raw_corr, -1.0, 1.0)

    #vectorized distance calculation
    dist_matrix = np.sqrt(2 * (1- safe_corr))

    return pd.DataFrame(
        dist_matrix,
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )

def compute_mst(distance_df):
    """
    Calculates the Minimum Spanning Tree (MST) from the distance matrix.
    This strips away market noise to extract the hierarchical skeleton.
    """
    dist_matrix = distance_df.values

    #calulate MST
    mst_sparse = minimum_spanning_tree(dist_matrix)

    #convert back into a grid for easy reading
    mst_dense = mst_sparse.toarray()

    return mst_dense

def extract_ultrametric(linkage_matrix, labels):
    """
    Computes the subdominant ultrametric distance between all nodes.
    By definition, this is the cophenetic distance of the single-linkage dendrogram.
    
    Arguments: linkage_matrix (ndarray): The (N-1) x 4 linkage matrix from execute_slc.
               labels (list/Index): The asset tickers/labels.
        
    Returns:
        pd.DataFrame: A symmetric DataFrame containing the ultrametric distances.
    """
    # cophenet() calculates the exact max-path distance in the tree
    # It returns a 1D array
    condensed_ultra = cophenet(linkage_matrix)
    
    # Convert back to a square N x N matrix
    ultra_matrix = squareform(condensed_ultra)
    
    return pd.DataFrame(ultra_matrix, index=labels, columns=labels)

def calculate_scaling_exponent(linkage_matrix, p=2):
    """
    Calculates the p-adic scaling exponent alpha_hat from linkage matrix

    returns alpha_hat and r_squared (goodness of fit)
    """

    #1. extract merge hieghts
    h = linkage_matrix[:, 2][::-1]
    h_1 = h[0]
    n_merges = len(h)

    #2. compute log height sequence lambda_ell = -log(h_ell / h_1)
    lambdas = - np.log(np.clip(h[1:] / h_1, 1e-10, 1.0))

    #3.define independent variable (depth of tree) x=(ell -1)
    x = np.arange(1, n_merges)

    #4. OLS regression through origin to find slope alpha_hat
    beta = np.sum(x * lambdas) / np.sum(x**2)
    alpha_hat = beta / np.log(p)

    #5. calculate R^2 for goodness of fit
    residuals = lambdas - (beta * x)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((lambdas - np.mean(lambdas))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot !=0 else 0

    return alpha_hat, r_squared
    

def execute_slc(distance_matrix):

    """
    Executes Single Linkage Clustering (equivalent to MST) and returns the linkage matrix for further analysis
    
    Arguments: distance_matrix (array): mategna distance matrix

    returns: linkage_matrix (array): hierarchical linkage matrix used for p-adic embeddings and alpha calculations.

    """

    # SciPy's linkage algorithm requires a 1D "condensed" distance array.
    # If your correlation_to_distance function outputs a standard 2D square matrix,
    # we need to condense it first to avoid SciPy throwing a dimensionality error.
    if len(distance_matrix.shape) == 2:
        # np.clip ensures no floating point errors push distances below 0
        dist_clipped = np.clip(distance_matrix, 0, None)
        condensed_dist = squareform(dist_clipped, checks=False)
    else:
        condensed_dist = distance_matrix
        
    # method='single' gives equivalence to the MST
    linkage_mat = linkage(condensed_dist, method='single')
    
    return linkage_mat