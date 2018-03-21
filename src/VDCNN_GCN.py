# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import numpy as np
import scipy, os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, constant

from src.graphconv.graph_utils import get_graph_laplacian, generate_word_embeddings
from src.pygcn.models import GCN

# np.set_printoptions(threshold='nan')
from src import lib
from src.dataset_utils import preprocess_data, mix_datasets, vectorize


class VDCNN_GCN(nn.Module):
    def __init__(self, opt, laplacian=None, laplacian_hidden = 16, n_classes=2, num_embedding=141,
                 embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=False, embeddings=None):
        super(VDCNN_GCN, self).__init__()

        layers = []
        fc_layers = []

        self.embed = nn.Embedding(num_embedding, embedding_dim, padding_idx=0, max_norm=None,
                                  norm_type=2, scale_grad_by_freq=False, sparse=False)
        #
        # self.gcn_embed = GCN(nfeat=embedding_dim * 2, nhid=laplacian_hidden, nout=embedding_dim, dropout=0.5)
        self.gcn_embed = GCN(nfeat=embeddings.shape[1], nhid=laplacian_hidden, nout=embedding_dim, dropout=0.5, embeddings=embeddings)
        self.L = laplacian
        #
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
        # print("L : ", self.L.shape)
        # print("x",x)
        # out = self.embed(x)
        # print(self.embed.weight.shape)
        # print(out.shape)
        out = self.gcn_embed(self.L)
        self.embed.weight = nn.Parameter(out.data)
        out = self.embed(x)
        out = out.transpose(1, 2)

        out = self.layers(out)

        out = out.view(out.size(0), -1)

        out = self.fc_layers(out)

        return out


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
    # TODO: Why is this required? What happens if we don't do this?
    assert np.std([x.shape[0] for x in arrays]) == 0

    for j in range(0, len(arrays[0]), batch_size):
        yield [x[j: j + batch_size] for x in arrays]


# def compare_embeddings(x, y):
#     # print(type(y.numpy()))
#     if x is not None:
#         tx = x.numpy()
#         ty = y.numpy()
#         diff = (float) (np.linalg.norm(tx, ord='fro') - np.linalg.norm(ty, ord='fro'))
#         print(diff)
#         # print(x.shape, y.shape)
#
#     return y

def train(opt, logger, model, criterion, tr_data, val_data, te_data, n_classes, dataset_name):
    lr = opt.lr
    if (opt.transfer_weights):
        lr = opt.transfer_lr

    logger.info("Setting the lr to : {}".format(lr))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
            # model.embed.weight.requires_grad = True
            # prev_matrix= compare_embeddings(prev_matrix, model.embed.weight.data)
            # xte, yte = te_data
            x_val, y_val = val_data
            val_gen = batchify([x_val, y_val], batch_size=opt.batch_size)
            y_prob = predict_from_model(val_gen, model, gpu=opt.gpu)
            val_metrics = lib.get_metrics(y_val, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])
            params = [dataset_name, n_iter, opt.iterations, tr_metrics, val_metrics]
            logger.info('{} - Iter [{}/{}]: train-metrics- {} ; val-metrics- {}'.format(*params))
            # val_params = [dataset_name, n_iter, opt.iterations, val_metrics]
            # logger.info('{} - Iter [{}/{}] - val-metrics: {}'.format(*val_params))

            diclogs = {
                "predictions": {
                    "test": {
                        "y_true": y_val,
                        "y_prob": y_prob
                    }
                },
                "name": "VDCNN",
                "parameters": vars(opt)
            }

            import pickle
            filename = "diclog_[{}|{}]_loss[{:.3f}|{:.3f}]_acc[{:.3f}|{:.3f}].pkl".format(n_iter, opt.iterations,
                                                                                          tr_metrics['logloss'],
                                                                                          val_metrics['logloss'],
                                                                                          tr_metrics['accuracy'],
                                                                                          val_metrics['accuracy'])

            with open('{}/{}'.format(opt.model_folder, filename), 'wb') as f:
                pickle.dump(diclogs, f, protocol=4)

            # Addition to save model and optimizer state dict
            model_dict = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }
            if best_accuracy < float(val_metrics['accuracy']):
                best_accuracy = float(val_metrics['accuracy'])
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


def test(model, logger, opt, te_data, n_classes, dataset_name):
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
    root_te_sentences, root_te_labels, transfer_te_sentences, transfer_te_labels, total_classes = mix_datasets(opt,
                                                                                                               logger)

    ## preprocess
    logger.info(" Joint Training: Txt vectorization...")
    n_txt_feats, tr_data, val_data, te_data, root_te_data, transfer_te_data, _ = \
        vectorize(opt, mixed_data_tr_sentences, mixed_data_label, mix_data_te_sentences, mix_data_te_labels,
                  root_te_sentences, root_te_labels, transfer_te_sentences, transfer_te_labels)
    logger.info("n_txt_feats before overriding: ", n_txt_feats)
    n_classes = int(np.max(mixed_data_label) + 1)
    logger.info("Number of classes in the mixed dataset are: ", n_classes, type(n_classes))
    ## construct model
    torch.manual_seed(opt.seed)
    logger.info("Seed for random numbers: ", torch.initial_seed())
    if (opt.num_embedding_features != -1):
        n_txt_feats = opt.num_embedding_features
        logger.info("Overriding the number of embedding features to: ", n_txt_feats)
    model = VDCNN_GCN(opt=opt, n_classes=total_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
                  n_fc_neurons=2048, shortcut=opt.shortcut)

    if opt.gpu:
        model.cuda()

    criterion = get_criterion(opt)

    dataset_name = "Mixed_" + opt.combined_datasets + "_" + str(opt.joint_ratio)
    if opt.joint_test == 1:
        logger.info("Testing on root dataset only")
        test(model, logger, opt, root_te_data, n_classes, dataset_name)
    elif opt.joint_test == 2:
        logger.info("Testing on transfer dataset only")
        test(model, logger, opt, transfer_te_data, n_classes, dataset_name)
    elif opt.joint_test == 3:
        logger.info("Testing on both datasets only")
        test(model, logger, opt, te_data, n_classes, dataset_name)
    else:
        logger.info("Joint training")
        train(opt, logger, model, criterion, tr_data, val_data, te_data, n_classes, dataset_name)

        logger.info("After Training: Testing on root dataset only")
        test(model, logger, opt, root_te_data, n_classes, dataset_name)

        logger.info("After Training: Testing on transfer dataset only")
        test(model, logger, opt, transfer_te_data, n_classes, dataset_name)

        logger.info("After Training: Testing on both datasets only")
        test(model, logger, opt, te_data, n_classes, dataset_name)


## Use this to transfer weights from pre-trained layers and run a new model
def transfer_and_train(opt, logger):
    # load the new data set:
    tr_data, val_data, te_data, n_classes, n_txt_feats, dataset_name, _ = preprocess_data(opt, logger)

    # define the structure of the model to be loaded - get most of the structure from the user, using input args:
    num_previous_classes = opt.num_prev_classes
    num_embeddings = opt.num_embedding_features

    pretrained_model = VDCNN_GCN(opt=opt, n_classes=num_previous_classes, num_embedding=num_embeddings, embedding_dim=16,
                             depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)

    # load the previously trained model
    checkpoint = torch.load(opt.model_load_path)
    pretrained_model.load_state_dict(checkpoint['model'])

    # Construct the new model:
    new_model = VDCNN_GCN(opt=opt, n_classes=n_classes, num_embedding=num_embeddings, embedding_dim=16,
                      depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)

    new_model = model_load_previous_structure(pretrained_model, new_model, opt.freeze_pre_trained_layers)
    if (opt.gpu):
        new_model.cuda()

    logger.info("New model loaded successfully, going to train")
    criterion = get_criterion(opt)

    train(opt, logger, new_model, criterion, tr_data, val_data, te_data, n_classes, dataset_name)
    test(new_model, logger, opt, te_data, n_classes, dataset_name)


def get_criterion(opt):
    if opt.class_weights:
        criterion = nn.CrossEntropyLoss(torch.cuda.FloatTensor(opt.class_weights))
        return criterion
    else:
        criterion = nn.CrossEntropyLoss()
        return criterion


def model_load_previous_structure(old_model, new_model, freeze_pre_trained_layers):
    new_model_dict = new_model.state_dict()
    pre_trained_dict = {k: v for k, v in old_model.state_dict().items() if "fc_layers" not in k}
    for k, v in pre_trained_dict.items():
        # print(k, type(v))
        v.requires_grad = False
    new_model_dict.update(pre_trained_dict)

    new_model.load_state_dict(new_model_dict)
    # print(list(new_model.parameters()))
    return new_model

def graph_convolution(opt, logger):
    # print(os.getcwd())
    names = [opt.dataset]
    ng_w2v = generate_word_embeddings(names)
    tr_data, val_data, te_data, n_classes, n_txt_feats, dataset_name, w2v_word_to_idx = preprocess_data(opt, logger, w2v=ng_w2v)
    L, A, embeddings = get_graph_laplacian(names, w2v=ng_w2v, txt_feature_size=n_txt_feats, w2v_word_to_idx=w2v_word_to_idx)
    scipy.sparse.save_npz("pygcn/sparse_matrix.npz", L[0])
    # scipy.sparse.save_npz("sparse_matrix.npz", L[0])
    # L[0] = scipy.sparse.load_npz("sparse_matrix.npz")
    L[0] = scipy.sparse.load_npz("pygcn/sparse_matrix.npz")


    print("n_txt_feats:", n_txt_feats)
    torch.manual_seed(opt.seed)
    print("Seed for random numbers: ", torch.initial_seed())

    if (opt.num_embedding_features != -1):
        n_txt_feats = opt.num_embedding_features
        logger.info("Overriding the number of embedding features to: ", n_txt_feats)

    model = VDCNN_GCN(opt, n_classes=n_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
                  n_fc_neurons=2048, shortcut=opt.shortcut, laplacian=L[0], embeddings=embeddings)

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