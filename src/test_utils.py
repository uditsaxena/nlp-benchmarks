import csv
import sys
import torch

from src.VDCNN import VDCNN

csv.field_size_limit(sys.maxsize)


def print_dataset(data, print_label, shape=False, limit=2):
    print("---")
    print("Printing {}".format(print_label))
    for i in range(limit):
        if (shape == False):
            print("{}: val: {}, label: {}".format(print_label, data[0][i], data[1][i]))
        else:
            print("{}: val: {}, shape: {}, label: {}".format(print_label, data[0][i], len(data[0][i]), data[1][i]))
    print("----")


def test_newsgroup20():
    # get_newsgroup_20()
    filename = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/datasets/20_newsgroups/newsgroup20_train.txt"
    # f = open(filename)
    f = open(filename, mode='r', encoding='utf-8')
    reader = csv.DictReader(f, fieldnames=['sub-c', 'categ', 'title', 'sender', 'description'], quotechar='"',
                            delimiter='\t')
    length = 0
    class_map = {}
    count = 1
    for line in reader:

        subc = line['sub-c']
        if (subc not in class_map.keys()):
            class_map[subc] = count
            print("ADDING {}".format(count))
            count += 1
            # else:
            # print("FOUND!!!")
            # print(subc, class_map[subc])
        # if length < 100:
        #     break
        length -= 1
    print(class_map)
    # line_length_cur = len(line.split("\t"))
    # if length != line_length_cur:
    #     print(line_length_cur)


def model_structure():
    model_path = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/models/VDCNN/AgNews_5000_model.pt"
    # model_path = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/models/VDCNN/ag_news-2000_model.pt"
    # model_path = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/models/VDCNN/ng20-2000_model.pt"
    print("Loooking at model now")
    n_classes=4
    #
    # model = VDCNN(n_classes=n_classes, num_embedding=n_txt_feats, embedding_dim=16, depth=opt.depth,
    #               n_fc_neurons=2048, shortcut=opt.shortcut)
    checkpoint = torch.load(model_path)['model']
    # print(checkpoint)
    for k,v in checkpoint.items():
        print(k, v.size())
    # model.load_state_dict(checkpoint)


    # print(list(model.children())[:-1])



if __name__ == '__main__':
    # test_newsgroup20()

    model_structure()
