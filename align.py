import math

import torch

# Small module for implementing mutual nearest neighbor batch correction. Import and
# use the align_query function to align the query dataset with a reference dataset using
# Euclidean distances in the raw feature space.

def get_mutual_neighbors(reference, query, neighborhood_size, ref_batch_size = 128, query_batch_size = 128):
    # Calculate cross-dataset mutual nearest neighbors between reference and query datasets using
    # Euclidean distance. Reference and query must both have shape (samples, features), but the size of the
    # sample axis can differ. The neighborhood_size argument determines the number of neighbors within which
    # to search for mutual neighbors. The returned array is a (pairs, 2)-shaped integer array containing
    # mutual pair indices, reference indices given in the first column and query indices in the second 
    # column.

    (neighbors_ref, neighbors_query) = ([], [])
    params = [(reference, query, neighbors_ref, ref_batch_size), 
              (query, reference, neighbors_query, query_batch_size)]
    for (data_1, data_2, results, batch_size) in params:
        num_batches = math.ceil(len(data_1) / batch_size)
        for (start_i, end_i) in ((i*batch_size, (i+1)*batch_size) for i in range(num_batches)):
            print(f"Running {start_i+1}/{len(data_1)}            ", end = "\r")
            dist = torch.square(data_1[start_i:end_i, None] - data_2[None]).sum(-1)
            neighbors = torch.argsort(dist, -1)[:, :neighborhood_size]
            results.append(neighbors)
    (neighbors_ref, neighbors_query) = (torch.cat(neighbors_ref), torch.cat(neighbors_query))
    neighbor_range = torch.arange(len(neighbors_ref), device = neighbors_ref.device)
    is_shared = (neighbors_query[neighbors_ref] == neighbor_range[:, None, None]).any(-1)
    ref_indices = torch.repeat_interleave(neighbor_range, neighborhood_size)
    paired = torch.stack([ref_indices, neighbors_ref.flatten()], -1)
    mutual_nn = paired[is_shared.flatten()]
    return mutual_nn

def get_correction(reference, query, mutual_neighbors, num_anchors, kernel_scale = 1.0, batch_size = 128):
    # Compute the additive correction that will be used to align the query dataset. Reference and query should
    # have the same sample ordering as in get_mutual_neigbors, but can use a different set of (shared) features.
    # The mutual_neighbors argument is the output from get_mutual_neighbors, while num_anchors determines how
    # many mutual pairs should be used to compute the correction for each query sample. The kernel_scale
    # parameter determines how aggressively the pair weighting should decay with distance from the sample.
    # The returned tuple contains the array of query corrections along with an array containing the indices
    # of the query anchors that were used to compute the correction for each sample. 

    (corrections, anchor_indices) = ([], [])
    for (start_i, end_i) in ((i*batch_size, (i+1)*batch_size) for i in range(math.ceil(len(query) / batch_size))):
        print(f"Running {start_i+1}/{len(query)}             ", end = "\r")
        ref_anchors = reference[mutual_neighbors[:, 0]]
        query_anchors = query[mutual_neighbors[:, 1]]
        query_dist = torch.square(query[start_i:end_i, None] - query_anchors[None]).sum(-1)**0.5
        nearest_anchors_index = torch.argsort(query_dist, -1)[:, :num_anchors]
        nearest_dists = query_dist[torch.arange(query_dist.shape[0])[:, None], nearest_anchors_index]
        dists_centered = nearest_dists - nearest_dists.amin(-1, keepdim = True)
        kernel = torch.exp(-dists_centered/kernel_scale)
        weights = kernel / kernel.sum(-1, keepdim = True)
        anchor_diff = ref_anchors[nearest_anchors_index] - query_anchors[nearest_anchors_index]
        correction = torch.einsum("nk,nkx->nx", weights, anchor_diff)
        corrections.append(correction)
        anchor_indices.append(nearest_anchors_index)
    (corrections, anchor_indices) = (torch.cat(corrections, 0), torch.cat(anchor_indices, 0))
    return (corrections, anchor_indices)

def align_query(reference, query, neighborhood_size = 5, ref_batch_size = 128, query_batch_size = 128, 
                num_anchors = 100, kernel_scale = 1.0):
    # Small convenience function to compute query correction. All arguments are directly passed to either the 
    # get_mutual_neighbors or get_correction functions. Returns the aligned query datatset.

    print("Determining neighborhood...")
    mutual_neighbors = get_mutual_neighbors(reference, query, neighborhood_size, ref_batch_size, query_batch_size)
    print("Computing correction...")
    (correction, nearest_anchors) = get_correction(reference, query, mutual_neighbors, num_anchors, kernel_scale, 
                                                   query_batch_size)
    query_corr = query + correction
    print("Completed.                           ")
    return query_corr