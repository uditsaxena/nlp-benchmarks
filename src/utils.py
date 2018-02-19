import os

from chazutsu.datasets import news_group20

def print_dataset(data, print_label, shape=False, limit = 2):
    print("---")
    print("Printing {}".format(print_label))
    for i in range(limit):
        if (shape == False):
            print("{}: val: {}, label: {}".format(print_label, data[0][i], data[1][i]))
        else:
            print("{}: val: {}, shape: {}, label: {}".format(print_label, data[0][i], len(data[0][i]), data[1][i]))
    print("----")

def get_newsgroup_20():
    dataset = news_group20.NewsGroup20()
    dataset.download(directory="/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/datasets")

if __name__ == '__main__':
    # get_newsgroup_20()
    f = open("/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/datasets/20_newsgroups/newsgroup20_test.txt")
    lines = f.readlines()
    print(lines[0].split("\t"))

