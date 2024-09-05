import torch
import glob
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from pathlib import Path



def read_data(filepath):
    assert isinstance(filepath, (str, Path))
    filepath = Path(filepath)
    filename, suffix = filepath.stem, filepath.suffix

    # check cache
    parent_dir = filepath.absolute().parent
    processed_fp = parent_dir / f"{filename}_processed.pt"
    if processed_fp.exists():
        data, slices = torch.load(processed_fp)
        return data, slices

    # no cached data
    print("processing data...")

    all_mol_datas = []

    for split in ['train', 'val', 'test']:
        f = filepath / f"{split}.pt"
        print(f)
        mol_datas = torch.load(f)
        all_mol_datas += mol_datas


    all_datas = []
    for graph in tqdm(all_mol_datas):
        x = graph[0]
        # if len(x) <= 300 :
        edge_attr = graph[1]
        edge_index = graph[2]
        y = graph[3]
        data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y)

        all_datas.append(data)
    data, slices = InMemoryDataset.collate(all_datas)

    torch.save((data, slices), processed_fp)
    print(f"saved to {processed_fp.absolute()}")

    return data, slices

class LRGBDataset(InMemoryDataset):
    def __init__(self, filepath):
        super(LRGBDataset, self).__init__()
        print("LRGBDataset: ===init===")
        self.data, self.slices = read_data(filepath)
        print(f"Dataset complete.")


