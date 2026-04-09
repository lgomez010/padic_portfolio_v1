import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree

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

def extract_ultrametric(mst_matrix, labels):
    """
    walks the MST to get the subdominant ultrametricc distance between all nodes.
    The ultrametric distance is the max edge weight on the path between two nodes
    """
    # turn our matrix into a network graph
    G = nx.from_numpy_array(mst_matrix)
    num_nodes = len(G.nodes)

    # create blank grid to hold new distances
    ultrametric_grid = np.zeros((num_nodes, num_nodes))

    # check every possible pair of stocks i and j
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):

            #find unique path between i and j in the tree
            path = nx.shortest_path(G, source = i, target = j)

            # find edge with max weight on this path
            max_weight = 0
            for k in range(len(path) -1):
                node_a = path[k]
                node_b = path[k+1]
                edge_weight = G[node_a][node_b]['weight']

                if edge_weight > max_weight:
                    max_weight = edge_weight

            #save max weight to out grid
            ultrametric_grid[i,j] = max_weight
            ultrametric_grid[j,i] = max_weight

    return pd.DataFrame(ultrametric_grid, index=labels, columns=labels)

    