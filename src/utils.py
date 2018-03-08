import csv
import os
import json
import sys

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


def read_lexalytics_dataset():
    filename = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/datasets/lex1/raw"

    f = open(filename, 'r', encoding='utf-8')
    reader = csv.DictReader(f)
    for line in reader:
        print("A:", line)
        # print("A:", line.keys())
    lines = f.readlines()
    count = 1
    #
    classlist = []
    for line in lines:
        json_str = json.loads(line)
        # print(json_str.keys())
        text = json_str['text']

        text = text.replace("\r\n"," ")
        classmap = json_str['analyses'][0]['worlds']
        # print(text)
        # if (len(classmap.keys()) == 1):
        #     print(classmap.keys())
        # for k, v in classmap.items():
            # classlist.append(k)
            # print(k)

        # count += 1
        # if count > 1:
        #     break
    classlist = list(set(classlist))
    # print(classlist)

    # print(len(lines))
    filename = "/Users/Udit/programs/github/S18_Code/lex/code/vdcnn/datasets/lex1/classmap.json"
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()
    c = 1
    class_mapping = {}
    for line in lines:
        newline = line.replace("[", "").replace("]", "")[:-1].split(',')[1]
        newline = newline.replace("\"", "")
        if (newline not in class_mapping.keys()):
            class_mapping[newline] = c
            c += 1
    print(class_mapping)





if __name__ == '__main__':
    read_lexalytics_dataset()
    # test_newsgroup20()
