import numpy as np

from src import lib
from src.datasets import load_datasets


def divide_data_using_ratio(tr_sentences, tr_labels, te_sentences, te_labels, ratio):
    data_size = len(list(tr_sentences))
    transfer_size = int(round(ratio * data_size))
    print("Transfer Ratio Size: {}".format(transfer_size))

    new_tr_sentences, new_tr_labels = [], []
    for i in range(transfer_size):
        new_tr_sentences.append(tr_sentences[i])
        new_tr_labels.append(tr_labels[i])

    te_sentences.extend(tr_sentences[:transfer_size])
    te_labels.extend(tr_labels[:transfer_size])

    return new_tr_sentences, new_tr_labels, te_sentences, te_labels


''' 
Load and preprocess one set of data, taken from opt.dataset
If 'test' is True, load the dataset going to be used for testing from opt.test_dataset 
Mainly used for training on one dataset and testing on the other
'''


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

    if (opt.target_transfer_ratio != 0.0):
        tr_sentences, tr_labels, te_sentences, te_labels = divide_data_using_ratio(tr_sentences, tr_labels,
                                                                                   te_sentences, te_labels,
                                                                                   opt.target_transfer_ratio)

    if opt.shuffle:
        logger.info("  - shuffling...")
        tr_sentences, tr_labels = utils.shuffle(tr_sentences, tr_labels, random_state=opt.seed)
        te_sentences, te_labels = utils.shuffle(te_sentences, te_labels, random_state=opt.seed)

    logger.info("  - txt vectorization...")
    n_txt_feats, te_data, tr_data = vectorize(opt, tr_sentences, tr_labels, te_sentences, te_labels)

    return tr_data, te_data, n_classes, n_txt_feats, dataset_name


def vectorize(opt, tr_sentences, tr_labels, te_sentences, te_labels):
    vec = lib.StringToSequence(level="char")
    vec.fit(tr_sentences)
    x_tr = vec.fit_transform(tr_sentences)
    x_tr = np.array(lib.pad_sequence(x_tr, maxlen=opt.maxlen, padding='post', truncating='post', value=0))
    x_te = vec.transform(te_sentences)
    x_te = np.array(lib.pad_sequence(x_te, maxlen=opt.maxlen, padding='post', truncating='post', value=0))
    n_txt_feats = int(max(x_tr.max(), x_te.max()) + 10)

    tr_data = [x_tr, np.array(tr_labels)]
    te_data = [x_te, np.array(te_labels)]
    return n_txt_feats, tr_data, te_data


def mix_data_using_ratio(root_data, transfer_data, joint_ratio):
    root_sentences, root_labels = lib.create_dataset(root_data, subsample_count=0)
    max_root_label_separation = np.max(root_labels) + 1

    # calculate data size to borrow

    transfer_sentences, transfer_labels = lib.create_dataset(transfer_data, subsample_count=0,
                                                             base_label=max_root_label_separation)
    transfer_data_size = len(list(transfer_sentences))
    print("Transfer Data Size: {}".format(transfer_data_size))
    transfer_size = int(round(joint_ratio * transfer_data_size))
    print("Transfer Ratio Size: {}".format(transfer_size))

    mixed_data_train, mixed_data_label = [], []
    mixed_data_train.extend(root_sentences)
    mixed_data_label.extend(root_labels)

    for i in range(transfer_size):
        mixed_data_train.append(transfer_sentences[i])
        mixed_data_label.append(transfer_labels[i])

    unused_transfer_sentences, unused_transfer_labels = transfer_sentences[transfer_size:], transfer_labels[
                                                                                            transfer_size:]

    # print("Root dataset length - sentences {}, labels {}".format(len(root_sentences), len(root_labels)))
    # print("Transfer dataset length - sentences {}, labels {}".format(len(transfer_sentences), len(transfer_labels)))
    # print("Transfer Size: ", transfer_size)
    # print("Mix dataset length - sentences {}, labels {}".format(len(mixed_data_train), len(mixed_data_label)))
    # print("Unused dataset length - sentences {}, labels {}".format(len(unused_transfer_sentences),
    #                                                                len(unused_transfer_labels)))

    return mixed_data_train, mixed_data_label, unused_transfer_sentences, unused_transfer_labels


def mix_datasets(opt, logger):
    joint_ratio = opt.joint_ratio

    datasets = opt.combined_datasets.split(',')
    root_dataset = datasets[0]
    transfer_dataset = datasets[1]

    root_tr_data, root_te_data = load_train_test_raw_dataset(root_dataset, logger)

    transfer_tr_data, transfer_te_data = load_train_test_raw_dataset(transfer_dataset, logger)

    logger.info("Both datasets loaded, going to mix ...")

    mixed_data_tr_sentences, mixed_data_label, unused_transfer_sentences, unused_transfer_labels = \
        mix_data_using_ratio(root_tr_data, transfer_tr_data, joint_ratio)

    root_te_sentences, root_te_labels = lib.create_dataset(root_te_data, subsample_count=0)
    max_label_separation = np.max(root_te_labels) + 1
    transfer_te_sentences, transfer_te_labels = lib.create_dataset(transfer_te_data, subsample_count=0,
                                                                   base_label=max_label_separation)

    mixed_te_sentences = list(root_te_sentences)
    mixed_te_labels = list(root_te_labels)

    mixed_te_sentences.extend(transfer_te_sentences)
    mixed_te_labels.extend(transfer_te_labels)

    mixed_te_sentences.extend(unused_transfer_sentences)
    mixed_te_labels.extend(unused_transfer_labels)

    print("Root dataset Test length - sentences {}, labels {}".format(len(root_te_sentences), len(root_te_labels)))
    print("Transfer dataset Test length - sentences {}, labels {}".format(len(transfer_te_sentences),
                                                                          len(transfer_te_labels)))
    print("Unused dataset Test length - sentences {}, labels {}".format(len(unused_transfer_sentences),
                                                                        len(unused_transfer_labels)))
    print("Mixed dataset Test length - sentences {}, labels {}".format(len(mixed_te_sentences), len(mixed_te_labels)))
    print("Mixed dataset Train length - sentences {}, labels {}".format(len(mixed_data_tr_sentences),
                                                                        len(mixed_data_label)))

    return mixed_data_tr_sentences, mixed_data_label, root_te_sentences, root_te_labels


def load_train_test_raw_dataset(dataset_name, logger):
    dataset = load_datasets(names=[dataset_name])[0]
    dataset_class_name = dataset.__class__.__name__
    n_classes = dataset.n_classes

    logger.info("dataset: {}, n_classes: {}".format(dataset_class_name, n_classes))
    logger.info("  - loading dataset...")

    tr_data = dataset.load_train_data()
    te_data = dataset.load_test_data()

    return tr_data, te_data
