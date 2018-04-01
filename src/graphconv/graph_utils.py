import time

import nltk
import numpy as np
import os
import scipy
from gensim import models

from src.datasets import load_datasets
from src.graphconv import graph, coarsening


def generate_word_embeddings(datasets):
    sentences = []
    for dataset in datasets:
        sentences = get_sentences_from_dataset(dataset, sentences)
    print("Starting vectorization")
    w2v = models.Word2Vec(sentences, max_vocab_size=100000)
    print("Vectorization done")
    return w2v


def get_sentences_from_dataset(dataset, sentences):
    train_sentences, train_labels, test_sentences, test_labels = get_sentences(dataset)
    for train_sentence in train_sentences:
        for sentence in nltk.sent_tokenize(train_sentence):
            sentences.append(nltk.word_tokenize(sentence))
    for test_sentence in test_sentences:
        for sentence in nltk.sent_tokenize(test_sentence):
            sentences.append(nltk.word_tokenize(sentence))
    print("Tokenization done")
    return sentences


def get_sentences(name):
    print("name: {}".format(name))
    dataset = load_datasets(names=[name])[0]
    # train data generator
    tr_gen = dataset.load_train_data(chunk_size=512)
    tr_sentences, train_labels = [], []
    for x, y in tr_gen:
        tr_sentences.extend(x)
        train_labels.extend(y)
    print(" train: (samples/labels) = ({}/{})".format(len(tr_sentences), len(train_labels)))
    # test data generator
    tr_gen = dataset.load_test_data(chunk_size=2048)
    te_sentences, test_labels = [], []
    for x, y in tr_gen:
        te_sentences.extend(x)
        test_labels.extend(y)
    print(" test: (samples/labels) = ({}/{})".format(len(te_sentences), len(test_labels)))

    return tr_sentences, train_labels, te_sentences, test_labels


def get_graph_laplacian(names, w2v=None, txt_feature_size=None, w2v_word_to_idx=None):
    vocab = w2v.wv.vocab
    print("Vocab size: ", len(vocab))
    number_edges = 2
    coarsening_levels = 0

    embeddings = None
    dataset_name = names[0]
    src_dir = "pygcn/"
    embeddings_path = src_dir + dataset_name + "_embeddings.npy"
    if os.path.isfile(embeddings_path):
        print("Loading embeddings...")
        embeddings = np.load(embeddings_path)
        print("Loaded embeddings... continuing on")
        print(type(embeddings))
    else:
        print("constructing embeddings...")
        embeddings = np.random.randn(txt_feature_size, w2v.vector_size)
        count = 0
        for word in vocab.keys():
            if word in w2v_word_to_idx:
                idx = w2v_word_to_idx[word.lower()]
                embeddings[idx, :] = w2v[word]
            else:
                count += 1
        print("Didn't find: ", count, " words out of ", txt_feature_size, " words.")
        embeddings = np.array(embeddings)
        print("Saving embeddings...")
        np.save(src_dir + dataset_name + "_embeddings", embeddings)
    L = None

    graph_path = src_dir + "graph_" + dataset_name + ".npz"
    if os.path.isfile(graph_path):
        L = scipy.sparse.load_npz(graph_path)
        print(L.shape)
    # Todo: if this does not happen, write the else of this loop
    else:
        t_start = time.process_time()
        print("Embedded, starting graph construction")
        dist, idx = graph.distance_lshforest(embeddings, k=number_edges, metric='cosine')
        print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
        A = graph.adjacency(dist, idx)
        # print(A)
        # print("{} > {} edges".format(A.nnz // 2, number_edges * graph_data.shape[0] // 2))
        A_random = graph.replace_random_edges(A, 0)
        graphs, perm = coarsening.coarsen(A_random, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        print(L[0].size)
        print(type(L[0]))
        # L[0] is a sparse csr Matrix
        L = L[0]
        scipy.sparse.save_npz(graph_path, L)
        print("Done with the graph")
    return L, embeddings
    # return None


if __name__ == '__main__':
    names = ['ng20_tiny']

    get_graph_laplacian(names)
