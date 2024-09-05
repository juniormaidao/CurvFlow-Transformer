# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist
import json, bz2
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.structure import Structure
import warnings
import numpy as np
# import pbc
from pymatgen.core.periodic_table import Element
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SPCL
import torch

class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "G:\Python37\Graphormer-main\examples\customized_dataset\dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root,subset=True)
            with open(r'L:\cf\zinc_train_subset.json', 'r') as json_file:
                train_set_AS = json.load(json_file)
            with open(r'L:\cf\zinc_val_subset.json', 'r') as json_file:
                valid_set_AS = json.load(json_file)
            with open(r'L:\cf\zinc_test_subset.json', 'r') as json_file:
                test_set_AS = json.load(json_file)
            train_set = (MyZINC(root=root,subset=True, split="train"), train_set_AS)
            valid_set = (MyZINC(root=root,subset=True, split="val"), valid_set_AS)
            test_set = (MyZINC(root=root,subset=True, split="test"), test_set_AS)
        elif name == "dcgat":
            with bz2.open("dcgat_3_005.json.bz2") as fh:
                data = json.loads(fh.read().decode('utf-8'))
            inner_dataset = [ComputedStructureEntry.from_dict(i) for i in data["entries"]]
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
