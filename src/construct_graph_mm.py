import scipy
import torch
import numpy as np

from src.graphconv import graph, coarsening

if __name__ == '__main__':
    graph_data = np.load("pygcn/ng20_embeddings.npy")
    print("Loaded embeddings... continuing on")

    if torch.cuda.is_available():
        mat = torch.FloatTensor(graph_data).cuda()
        r = torch.mm(mat, mat.t())

    # # print(type(graph_data))
    # # t_start = time.process_time()
    # # print("Embedded, starting graph construction")
    # # number_edges = 16
    # coarsening_levels = 0
    # # dist, idx = graph.distance_sklearn_metrics(graph_data, k=number_edges, metric='cosine')
    # # print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # # A = graph.adjacency(dist, idx)
    # A = None
    # # print("{} > {} edges".format(A.nnz // 2, number_edges * graph_data.shape[0] // 2))
    # A_random = graph.replace_random_edges(A, 0)
    # graphs, perm = coarsening.coarsen(A_random, levels=coarsening_levels, self_connections=False)
    # L = [graph.laplacian(A, normalized=True) for A in graphs]
    # # print(L[0].shape)
    # # L[0] is a sparse csr Matrix
    # scipy.sparse.save_npz("pygcn/graph_ng20.npz", L[0])