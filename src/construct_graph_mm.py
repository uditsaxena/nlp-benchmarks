import numpy as np
import torch


def dist_calculation(tensor):
    # print(torch.mm(tensor, tensor.t()))
    size = tensor.shape
    print("Size : ", size)
    batch_size = int(size[0] / 64)
    rows = size[0]
    i = 0
    num_batches = int(rows / batch_size) + 1
    if rows % batch_size == 0:
        num_batches -= 1
    # print(tensor)
    num_edges = 3
    nn_list = []
    for _ in range(num_batches):
        print(i)
        begin = i
        end = begin + batch_size
        if end > rows:
            end = rows
        x_tensor = tensor[begin:end, :]
        curr_batch_mul_list = []
        tensor_batch_dist_list = []
        for k in range(batch_size):
            curr_tensor = x_tensor[k, :]
            j = 0
            curr_tensor_dist_list = []
            for _ in range(num_batches):
                # print(i, j)
                mul_begin = j
                mul_end = mul_begin + batch_size
                if mul_end > rows:
                    mul_end = rows
                y_tensor = tensor[mul_begin: mul_end, :]
                # print(y_tensor)
                dist = y_tensor - curr_tensor
                curr_tensor_batch_dist = torch.sum(torch.pow(dist, 2), dim=1)
                # print(curr_tensor_batch_dist)
                curr_tensor_dist_list.append(curr_tensor_batch_dist)
                j = mul_end
            curr_tensor_batch_dist_tensor = torch.cat(curr_tensor_dist_list, 0)
            # print(curr_tensor_batch_dist_tensor)
            tensor_batch_dist_list.append(curr_tensor_batch_dist_tensor.view(1, -1))
        # print("--")
        batch_mul_tensor = torch.cat(tensor_batch_dist_list, 0)
        # print(batch_mul_tensor)
        values, indices = torch.sort(batch_mul_tensor, 1, descending=True)
        nn_list.append(indices[:, 0:num_edges + 1])
        i = end
    dist_matrix = torch.cat(nn_list)
    print(dist_matrix.size())


if __name__ == '__main__':
    graph_data = np.load("pygcn/ng20_embeddings.npy")
    print("Loaded embeddings... continuing on")
    if torch.cuda.is_available():
        mat = torch.FloatTensor(graph_data).cuda()
        dist_calculation(mat)
    else:
        mat = torch.FloatTensor(graph_data)
        dist_calculation(mat)
