# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, constant

from src import lib
from src.dataset_utils import preprocess_data, mix_datasets, vectorize


def get_args():
    parser = argparse.ArgumentParser("""
    Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)
    """)
    parser.add_argument("--dataset", type=str, default='imdb')
    parser.add_argument("--test_dataset", type=str, default='ng20', help="The dataset to test on")
    parser.add_argument("--model_folder", type=str, default="models/VDCNN/imdb")
    parser.add_argument("--model_save_path", type=str, default="models/VDCNN/VDCNN_ag_news_depth@9")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=9,
                        help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train and test sets")
    parser.add_argument("--chunk_size", type=int, default=2048, help="number of examples read from disk")
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--test_batch_size", type=int, default=512,
                        help="number of example read by the gpu during test time")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=100,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--test_interval", type=int, default=50, help="Number of iterations between testing phases")
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=1337)
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
    parser.add_argument(("--joint_test"), type=int, default=0,help="0 for none, 1 for root, 2 for transfer, 3 for both")
    parser.add_argument(("--num_embedding_features"), type=int, default=-1, help="-1 for no use, otherwise use")

    args = parser.parse_args()
    return args


def predict_from_model(generator, model, gpu=True):
    model.eval()
    y_prob = []

    for data in generator:
        tdata = [Variable(torch.from_numpy(x).long(), volatile=True) for x in data]
        if gpu:
            tdata = [x.cuda() for x in tdata]

        yhat = model(tdata[0])

        # normalizing probs
        yhat = nn.functional.softmax(yhat)

        y_prob.append(yhat)

    y_prob = torch.cat(y_prob, 0)
    y_prob = y_prob.cpu().data.numpy()

    model.train()
    return y_prob


def batchify(arrays, batch_size=128):
    assert np.std([x.shape[0] for x in arrays]) == 0

    for j in range(0, len(arrays[0]), batch_size):
        yield [x[j: j + batch_size] for x in arrays]


class BasicConvResBlock(nn.Module):
    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsample=None):
        super(BasicConvResBlock, self).__init__()

        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out


class VDCNN(nn.Module):
    def __init__(self, n_classes=2, num_embedding=141, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=False):
        super(VDCNN, self).__init__()

        layers = []
        fc_layers = []

        self.embed = nn.Embedding(num_embedding, embedding_dim, padding_idx=0, max_norm=None,
                                  norm_type=2, scale_grad_by_freq=False, sparse=False)
        layers.append(nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

        layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        for _ in range(n_conv_block_64 - 1):
            layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
        layers.append(
            BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_128 - 1):
            layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        layers.append(
            BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_256 - 1):
            layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
        layers.append(
            BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_512 - 1):
            layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

        if opt.last_pooling_layer == 'k-max-pooling':
            layers.append(nn.AdaptiveMaxPool1d(8))
            fc_layers.extend([nn.Linear(8 * 512, n_fc_neurons), nn.ReLU()])
        elif opt.last_pooling_layer == 'max-pooling':
            layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
            fc_layers.extend([nn.Linear(61 * 512, n_fc_neurons), nn.ReLU()])
        else:
            raise

        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    constant(m.bias, 0)

    def forward(self, x):

        out = self.embed(x)
        out = out.transpose(1, 2)

        out = self.layers(out)

        out = out.view(out.size(0), -1)

        out = self.fc_layers(out)

        return out


def train(opt, model, criterion, tr_data, te_data, n_classes, dataset_name):
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    model.train()
    tr_gen = batchify(tr_data, batch_size=opt.batch_size)
    best_accuracy = -1
    for n_iter in range(opt.iterations):
        try:
            data = tr_gen.__next__()
        except StopIteration:
            tr_gen = batchify(tr_data, batch_size=opt.batch_size)
            data = tr_gen.__next__()

        tdata = [Variable(torch.from_numpy(x).long()) for x in data]
        if opt.gpu:
            tdata = [x.cuda() for x in tdata]

        tx, ty_true = tdata
        y_true = data[-1]

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        yhat = model(tx)
        y_prob = yhat.cpu().data.numpy()

        loss = criterion(yhat, ty_true)
        loss.backward()
        optimizer.step()

        tr_metrics = lib.get_metrics(y_true, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])

        params = [dataset_name, n_iter, opt.iterations, tr_metrics]
        logger.info('{} - Iter [{}/{}] - train metrics: {}'.format(*params))

        if n_iter % opt.test_interval == 0:
            xte, yte = te_data
            te_gen = batchify([xte, yte], batch_size=opt.batch_size)
            y_prob = predict_from_model(te_gen, model, gpu=opt.gpu)
            te_metrics = lib.get_metrics(yte, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])
            params = [dataset_name, n_iter, opt.iterations, tr_metrics, te_metrics]
            logger.info('{} - Iter [{}/{}] - train metrics: {}, test: {}'.format(*params))
            test_params = [dataset_name, n_iter, opt.iterations, te_metrics]
            logger.info('{} - Iter [{}/{}] - test-metrics: {}'.format(*test_params))

            diclogs = {
                "predictions": {
                    "test": {
                        "y_true": yte,
                        "y_prob": y_prob
                    }
                },
                "name": "VDCNN",
                "parameters": vars(opt)
            }

            import pickle
            filename = "diclog_[{}|{}]_loss[{:.3f}|{:.3f}]_acc[{:.3f}|{:.3f}].pkl".format(n_iter, opt.iterations,
                                                                                          tr_metrics['logloss'],
                                                                                          te_metrics['logloss'],
                                                                                          tr_metrics['accuracy'],
                                                                                          te_metrics['accuracy'])

            with open('{}/{}'.format(opt.model_folder, filename), 'wb') as f:
                pickle.dump(diclogs, f, protocol=4)

            # Addition to save model and optimizer state dict
            model_dict = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }
            if best_accuracy < float(te_metrics['accuracy']):
                best_accuracy = float(te_metrics['accuracy'])
                torch.save(model_dict, opt.model_save_path + "/{}".format("best") + "_model.pt")

            model_count = n_iter % (opt.test_interval * 5)
            model_name = dataset_name + "_" + str(model_count)
            torch.save(model_dict, opt.model_save_path + "/{}".format(model_name) + "_model.pt")

        if n_iter % opt.lr_halve_interval == 0 and n_iter > 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("new lr: {}".format(lr))


def test(model, te_data, n_classes, dataset_name):
    xte, yte = te_data
    te_gen = batchify([xte, yte], batch_size=opt.batch_size)
    checkpoint = torch.load(opt.model_load_path)
    model.load_state_dict(checkpoint['model'])

    # predict_from_model(te_gen, model, gpu=opt.gpu)
    y_prob = predict_from_model(te_gen, model, gpu=opt.gpu)
    te_metrics = lib.get_metrics(yte, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])
    params = [dataset_name, te_metrics]
    logger.info('{} - , test metrics: {}'.format(*params))


## Using arguments from opt, jointly train
def joint_train(opt, logger):
    ## get a mixed dataset
    mixed_data_tr_sentences, mixed_data_label, mix_data_te_sentences, mix_data_te_labels, \
    root_te_sentences, root_te_labels, transfer_te_sentences, transfer_te_labels = mix_datasets(opt, logger)

    ## preprocess
    logger.info("  - txt vectorization...")
    n_txt_feats, tr_data, te_data, root_te_data, transfer_te_data = \
        vectorize(opt, mixed_data_tr_sentences, mixed_data_label, mix_data_te_sentences, mix_data_te_labels,
                  root_te_sentences, root_te_labels, transfer_te_sentences, transfer_te_labels)

    n_classes = int(np.max(mixed_data_label) + 1)
    logger.info("Number of classes in the mixed dataset are: ", n_classes, type(n_classes))
    ## construct model
    torch.manual_seed(opt.seed)
    logger.info("Seed for random numbers: ", torch.initial_seed())
    if (opt.num_embedding_features != -1):
        n_txt_feats = opt.num_embedding_features
        logger.info("Overriding the number of embedding features to: ", n_txt_feats)
    model = VDCNN(n_classes=n_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
                  n_fc_neurons=2048, shortcut=opt.shortcut)

    if opt.gpu:
        model.cuda()

    if opt.class_weights:
        criterion = nn.CrossEntropyLoss(torch.cuda.FloatTensor(opt.class_weights))
    else:
        criterion = nn.CrossEntropyLoss()

    dataset_name = "Mixed_" + opt.combined_datasets + "_" + str(opt.joint_ratio)
    if opt.joint_test == 1:
        print("Testing on root dataset only")
        test(model, root_te_data, n_classes, dataset_name)
    if opt.joint_test == 2:
        print("Testing on transfer dataset only")
        test(model, transfer_te_data, n_classes, dataset_name)
    if opt.joint_test == 3:
        print("Testing on both datasets only")
        test(model, te_data, n_classes, dataset_name)
    else:
        train(opt, model, criterion, tr_data, te_data, n_classes, dataset_name)


if __name__ == "__main__":

    opt = get_args()

    if not os.path.exists(opt.model_folder):
        os.makedirs(opt.model_folder)

    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))

    # ut.print_dataset(tr_data, "train", True, 5)
    # ut.print_dataset(te_data, "test", True, 5)

    ## check if jointly training
    if (opt.joint_training):
        print("Joint Training !")
        joint_train(opt, logger)

    else:
        tr_data, te_data, n_classes, n_txt_feats, dataset_name = preprocess_data(opt, logger)
        test_tr_data, test_te_data, test_n_classes, test_n_txt_feats, test_dataset_name = preprocess_data(opt, logger,
                                                                                                          test=True)

        torch.manual_seed(opt.seed)
        print("Seed for random numbers: ", torch.initial_seed())

        model = VDCNN(n_classes=n_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
                      n_fc_neurons=2048, shortcut=opt.shortcut)

        if opt.gpu:
            model.cuda()

        if opt.class_weights:
            criterion = nn.CrossEntropyLoss(torch.cuda.FloatTensor(opt.class_weights))
        else:
            criterion = nn.CrossEntropyLoss()

        if opt.test_only == 1:
            print("Testing only")
            test(model, test_te_data, n_classes)
        else:
            print("Training...")
            train(opt, model, criterion, tr_data, te_data, n_classes, dataset_name)
