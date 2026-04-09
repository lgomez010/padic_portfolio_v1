import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def get_cluster_variance(cov_matrix, cluster_indices):
    """
    Calculate the total risk (variance) of a specific branch from the tree
    """
    cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
    inv_diag = 1.0 / np.diag(cluster_cov.values)
    weights = inv_diag / np.sum(inv_diag)
    return np.dot(weights, np.dot(cluster_cov.values, weights))

def allocate_hrp(ultrametric_df, returns_df):
    """
    takes topological distance and stock returns and outputs final portfolio weights
    """
    # calculate risk
    cov_matrix = returns_df.cov()

    #1. sort stocks (groups stocks that are topologically close)
    condensed_dist = squareform(ultrametric_df.values, checks=False)
    linkage_matrix = sch.linkage(condensed_dist, method='single')
    sort_ix = sch.leaves_list(linkage_matrix)

    #2. initialize weights 
    weights = pd.Series(1.0, index=sort_ix)

    """
    3. recursive bisection (dividing the loot)
    this splits into two groups and looks at the risk for
    each group. Calculates alpha (weight factor) and distributes 
    the weights accordingly. Then it repeats this process until we have
    assigned weights to all stocks.
    """
    
    queue = [(list(sort_ix), 1.0)]

    while len(queue) > 0:
        c, current_weight = queue.pop(0)

        if len(c) == 1:
            #reached a leaf node, assign weight
            weights[c[0]] = current_weight
            continue

        #bisect cluster
        mid = len(c) // 2
        c1 = c[:mid]
        c2 = c[mid:]

        #calculate risk for each cluster
        v1 = get_cluster_variance(cov_matrix, c1)
        v2 = get_cluster_variance(cov_matrix, c2)
        alpha = 1- v1 / (v1 + v2)

        #divide current weight between the two clusters 
        queue.append((c1, current_weight * alpha))
        queue.append((c2, current_weight * (1-alpha)))

    weights.index = ultrametric_df.index[weights.index]
    return weights.sort_values(ascending=False)