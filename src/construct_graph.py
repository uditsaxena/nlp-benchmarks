import scipy

from src import lib
from src.dataset_utils import preprocess_data
from src.graphconv.graph_utils import generate_word_embeddings, get_graph_laplacian
from src.main import get_args

if __name__ == '__main__':
    opt = get_args()

    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))
    names = [opt.dataset]
    ng_w2v = generate_word_embeddings(names)
    tr_data, val_data, te_data, n_classes, n_txt_feats, dataset_name, w2v_word_to_idx = preprocess_data(opt, logger,
                                                                                                        w2v=ng_w2v)
    L, A, embeddings = get_graph_laplacian(names, w2v=ng_w2v, txt_feature_size=n_txt_feats,
                                           w2v_word_to_idx=w2v_word_to_idx)
    scipy.sparse.save_npz("pygcn/sparse_matrix.npz", L[0])
    # scipy.sparse.save_npz("sparse_matrix.npz", L[0])
    # L[0] = scipy.sparse.load_npz("sparse_matrix.npz")
    L[0] = scipy.sparse.load_npz("pygcn/sparse_matrix.npz")
