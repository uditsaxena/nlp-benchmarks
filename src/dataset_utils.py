import numpy as np

from src import lib, utils
from src.datasets import load_datasets


## Mainly used for training on one dataset and testing on the other
##
def preprocess_data(opt, logger, test=False):
    dataset = load_datasets(names=[opt.dataset])[0]
    if (test == True):
        dataset = load_datasets(names=[opt.test_dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes

    logger.info("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    logger.info("  - loading dataset...")
    tr_data = dataset.load_train_data()
    te_data = dataset.load_test_data()

    logger.info("  - loading train samples...")
    tr_sentences, tr_labels = lib.create_dataset(tr_data, subsample_count=0)
    logger.info("  - loading train samples... {} samples".format(len(tr_sentences)))

    logger.info("  - loading test samples...")
    te_sentences, te_labels = lib.create_dataset(te_data, subsample_count=0)
    logger.info("  - loading test samples... {} samples".format(len(te_sentences)))

    if opt.shuffle:
        logger.info("  - shuffling...")
        tr_sentences, tr_labels = utils.shuffle(tr_sentences, tr_labels, random_state=opt.seed)
        te_sentences, te_labels = utils.shuffle(te_sentences, te_labels, random_state=opt.seed)

    logger.info("  - txt vectorization...")
    vec = lib.StringToSequence(level="char")
    vec.fit(tr_sentences)
    x_tr = vec.fit_transform(tr_sentences)
    x_tr = np.array(lib.pad_sequence(x_tr, maxlen=opt.maxlen, padding='post', truncating='post', value=0))
    x_te = vec.transform(te_sentences)
    x_te = np.array(lib.pad_sequence(x_te, maxlen=opt.maxlen, padding='post', truncating='post', value=0))
    n_txt_feats = int(max(x_tr.max(), x_te.max()) + 10)
    logger.info("  - txt train/test min/max: [{}|{}] [{}|{}]".format(x_tr.min(), x_tr.max(), x_te.min(), x_te.max()))

    tr_data = [x_tr, np.array(tr_labels)]
    te_data = [x_te, np.array(te_labels)]

    return tr_data, te_data, n_classes, n_txt_feats, dataset_name

def mix_datasets(opt, logger, ratio):
    datasets = opt.combined_datasets.split(',')
    root_dataset = datasets[0]
    transfer_dataset = datasets[1]

