import argparse

import scipy
import time, os
import numpy as np
from src import lib
from src.dataset_utils import preprocess_data
from src.graphconv import graph, coarsening
from src.graphconv.graph_utils import generate_word_embeddings, get_graph_laplacian


def get_args():
    parser = argparse.ArgumentParser("""
    Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)
    """)
    parser.add_argument("--dataset", type=str, default='ng20_tiny')
    parser.add_argument("--test_dataset", type=str, default='ng20', help="The dataset to test on")
    parser.add_argument("--model_folder", type=str, default="models/VDCNN/VDCNN_ng20_tiny_depth@9")
    parser.add_argument("--model_save_path", type=str, default="models/VDCNN/VDCNN_ng20_tiny_depth@9")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=9,
                        help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train and test sets")
    parser.add_argument("--chunk_size", type=int, default=2048, help="number of examples read from disk")
    parser.add_argument("--batch_size", type=int, default=32, help="number of example read by the gpu")
    parser.add_argument("--test_batch_size", type=int, default=512,
                        help="number of example read by the gpu during test time")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=100,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--test_interval", type=int, default=2, help="Number of iterations between testing phases")
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation_ratio", type=float, default=0.01)
    parser.add_argument("--last_pooling_layer", type=str, choices=['k-max-pooling', 'max-pooling'],
                        default='k-max-pooling', help="type of last pooling layer")

    parser.add_argument("--test_only", type=int, default=0, help="If you want to test only")
    parser.add_argument("--model_load_path", type=str, default="models/VDCNN/VDCNN_ag_news_depth@9",
                        help="Load pre-trained model from here")

    parser.add_argument("--combined_datasets", type=str, default="ag_news---ng20",
                        help="comma-sep list of two datasets, in the order - 'root,target' datasets")
    parser.add_argument("--joint_training", type=bool, default=False,
                        help="1 for joint training, 0 for no joint training")
    parser.add_argument("--joint_ratio", type=float, default=0.5,
                        help="Ratio of target to source dataset for joint training")
    parser.add_argument(("--joint_test"), type=int, default=0,
                        help="0 for none, 1 for root, 2 for transfer, 3 for both")
    parser.add_argument(("--num_embedding_features"), type=int, default=-1, help="-1 for no use, otherwise use")

    parser.add_argument("--transfer_weights", type=bool, default=False,
                        help="If true, transfer all except last fc-pre-trained layers")
    parser.add_argument("--num_prev_classes", type=int, default=20,
                        help="Number of classes in previously trained model")
    parser.add_argument("--transfer_lr", type=float, default=0.001, help="Used for fine tuning the final layer")
    parser.add_argument("--freeze_pre_trained_layers", type=bool, default=False,
                        help="Set to True if freezing previously trained layers")

    parser.add_argument("--gcn", type=bool, default=True,
                        help="If true, run a gcn type vdcnn's")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # opt = get_args()
    #
    # if not os.path.exists(opt.model_folder):
    #     os.makedirs(opt.model_folder)
    #
    # logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    # logger.info("parameters: {}".format(vars(opt)))
    #
    # names = [opt.dataset]
    # ng_w2v = generate_word_embeddings(names)
    # tr_data, val_data, te_data, n_classes, n_txt_feats, dataset_name, w2v_word_to_idx = preprocess_data(opt, logger,
    #                                                                                                     w2v=ng_w2v)
    # L, embeddings = get_graph_laplacian(names, w2v=ng_w2v, txt_feature_size=n_txt_feats,
    #                                     w2v_word_to_idx=w2v_word_to_idx)

    graph_data = np.load("pygcn/ng20_embeddings.npy")
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