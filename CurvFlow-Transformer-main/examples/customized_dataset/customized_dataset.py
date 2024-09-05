from examples.customized_dataset.scaffolds_split import scaffold_split
import json, bz2
from torch_geometric.datasets import *
from graphormer.data import register_dataset

@register_dataset("customized_qm9_dataset")

def create_customized_dataset():
    dataset = MoleculeNet(root=r'G:\Python37\Graphormer-main\examples\customized_dataset\dataset', name="bace")

    num_graphs = len(dataset)

    print(num_graphs)
    # customized datasetBAS  split
    train_idx, valid_idx, test_idx = scaffold_split(dataset, 0.1, 0.1, seed = 1)
    # random split
    # train_valid_idx, test_idx = train_test_split(
    #     np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    # )
    # train_idx, valid_idx = train_test_split(
    #     train_valid_idx, test_size=num_graphs // 5, random_state=0     bbbp1cf 0.70748
    # )
    print(train_idx.shape)
    print(valid_idx.shape)
    print(test_idx.shape)
    with open(r'L:\cf\bace_cf.json', 'r') as json_file:
        As = json.load(json_file)

    return {
        "dataset": (dataset, As),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(suppress=True)
    c = create_customized_dataset()
    print(c['test_idx'])