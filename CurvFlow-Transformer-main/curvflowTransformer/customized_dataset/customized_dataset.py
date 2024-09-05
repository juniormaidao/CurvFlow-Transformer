from examples.customized_dataset.LRGB import LRGBDataset
from examples.customized_dataset.scaffolds_split import scaffold_split
from graphormer.data import register_dataset
from torch_geometric.datasets import *
import json, bz2


@register_dataset("customized_dataset")
def create_customized_dataset():
    dataset = MoleculeNet(root='examples/dataset', name="bace")

    num_graphs = len(dataset)

    print(num_graphs)
    # customized datasetBAS  split
    train_idx, valid_idx, test_idx = scaffold_split(dataset, 0.1, 0.1, seed=1)

    print(train_idx.shape)
    print(test_idx.tolist())
    print(test_idx.shape)
    with open('curvflowTransformer/data/ma/bace_cf.json', 'r') as json_file:
        As = json.load(json_file)

    return {
        "dataset": (dataset, As),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
