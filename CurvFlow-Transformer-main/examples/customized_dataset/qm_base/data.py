import numpy as np
import torch
from dataset import QMCompDataset


def prepare_dataset(tr_filepath):
    # load data files
    tr_dataset = QMCompDataset(tr_filepath)
    print("train_dataset:", len(tr_dataset))

    target = "energy"
    tr_dataset.data["y"] = tr_dataset.data[target]
    tr_dataset.slices["y"] = tr_dataset.slices[target]

    # split validation set
    perm = torch.randperm(len(tr_dataset))
    train_num = int(0.8 * len(tr_dataset))
    train_idx = perm[:train_num]
    valid_idx = perm[train_num:]

    train_dataset = tr_dataset[train_idx]
    valid_dataset = tr_dataset[valid_idx]
    print("split dataset complete")
    print("tr:", len(train_dataset), "val:", len(valid_dataset))
    return train_dataset, valid_dataset


if __name__ == '__main__':
    train_dataset, valid_dataset = prepare_dataset(r"D:\BaiduNetdiskDownload\QMA_round1_train_230725_0.npy")
    for i in range(20):
        # print(train_dataset[i])
        print(train_dataset[i].keys)
        print('name:', train_dataset[i]['name'])
        print('atom_count:', train_dataset[i]['atom_count'])
        print('bond_count:', train_dataset[i]['bond_count'])
        print('z:', train_dataset[i]['z'])
        print('energy:', train_dataset[i]['energy'])
        print('force:', train_dataset[i]['force'])
        print('y:', train_dataset[i]['y'])
    # print('bond_count:', train_dataset[i]['bond_count'])
    # print('connectivity:', train_dataset[i]['connectivity'])
    # connectivity = train_dataset[i]['connectivity']
    # bound_count = int(train_dataset[i]['bond_count'])
    # A = [0 for i in range(bound_count)]
    # B = [0 for i in range(bound_count)]
    # for i in range(bound_count):
    #     A[i] = connectivity[i][0]
    #     B[i] = connectivity[i][1]
    # edge_index  = torch.tensor([A, B])
    # print(edge_index )

