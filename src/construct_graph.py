import scipy
import time
import numpy as np
import torch

from src.graphconv import graph, coarsening

def check_dim():
    # L = scipy.sparse.load_npz("pygcn/graph_ng20.npz")
    # print(L.size)
    L = scipy.sparse.load_npz("pygcn/graph_agnews.npz")
    print(L.size)
    print(type(L))
    print(type(L.tocoo()))
    X = L.tocoo()
    # print(X.data)
    # print(X.row)
    # print( X.col)
    adj = torch.autograd.Variable(torch.sparse.FloatTensor(X.data, (X.row, X.col)))

if __name__ == '__main__':
    check_dim()
    # graph_data = np.load("pygcn/ng20_embeddings.npy")
    # print("Loaded embeddings... continuing on")
    # print(type(graph_data))
    # t_start = time.process_time()
    # print("Embedded, starting graph construction")
    # number_edges = 16
    # coarsening_levels = 0
    # dist, idx = graph.distance_sklearn_metrics(graph_data, k=number_edges, metric='cosine')
    # print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # A = graph.adjacency(dist, idx)
    #
    # # print("{} > {} edges".format(A.nnz // 2, number_edges * graph_data.shape[0] // 2))
    # A_random = graph.replace_random_edges(A, 0)
    # graphs, perm = coarsening.coarsen(A_random, levels=coarsening_levels, self_connections=False)
    # L = [graph.laplacian(A, normalized=True) for A in graphs]
    # # print(L[0].shape)
    # # L[0] is a sparse csr Matrix
    # scipy.sparse.save_npz("pygcn/graph_ng20.npz", L[0])