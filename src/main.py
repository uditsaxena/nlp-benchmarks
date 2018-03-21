import argparse
import os

import torch

from src import lib
from src.VDCNN import transfer_and_train, joint_train, VDCNN, get_criterion, test, train
from src.VDCNN_GCN import graph_convolution
from src.dataset_utils import preprocess_data


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

    opt = get_args()

    if not os.path.exists(opt.model_folder):
        os.makedirs(opt.model_folder)

    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))

    # ut.print_dataset(tr_data, "train", True, 5)
    # ut.print_dataset(te_data, "test", True, 5)

    ## check if jointly training

    if opt.gcn:
        print("Graph convolutions")
        graph_convolution(opt, logger)

    elif (opt.transfer_weights):
        logger.info("Transfer weights from pre-trained layers")
        transfer_and_train(opt, logger)

    elif (opt.joint_training):
        logger.info("Joint Training !")
        joint_train(opt, logger)

    else:
        logger.info("Simple training")
        tr_data, val_data, te_data, n_classes, n_txt_feats, dataset_name, _ = preprocess_data(opt, logger)

        torch.manual_seed(opt.seed)
        print("Seed for random numbers: ", torch.initial_seed())

        if (opt.num_embedding_features != -1):
            n_txt_feats = opt.num_embedding_features
            logger.info("Overriding the number of embedding features to: ", n_txt_feats)

        model = VDCNN(opt,n_classes=n_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
                      n_fc_neurons=2048, shortcut=opt.shortcut)

        if opt.gpu:
            model.cuda()

        criterion = get_criterion(opt)

        if opt.test_only == 1:
            logger.info("Testing only")
            test_tr_data, test_val_data, test_te_data, test_n_classes, test_n_txt_feats, test_dataset_name, _ = \
                preprocess_data(opt, logger, test=True)
            test(model, logger, opt, test_te_data, n_classes, dataset_name)
        else:
            logger.info("Training...")
            train(opt, logger, model, criterion, tr_data, val_data, te_data, n_classes, dataset_name)

            logger.info("Testing...")
            test(model, logger, opt, te_data, n_classes, dataset_name)
