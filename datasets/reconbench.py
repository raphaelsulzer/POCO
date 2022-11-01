from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import glob
import logging

class Reconbench(Dataset):

    def __init__(self, root, scan, split="training", categories=None, transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, **kwargs):
            
        super().__init__(root, transform, None)

        logging.info(f"Dataset  - Reconbench- {dataset_size}")

        if categories is None:
            with open(os.path.join(root,"classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')

        self.root = root
        self.scan = scan
        self.split = split
        self.filter_name = filter_name
        self.filelists = []
        self.num_non_manifold_points = num_non_manifold_points


        self.filenames = categories


        logging.info(f"Dataset - len {len(self.filenames)}")


        self.metadata = None

    def get_category(self, f_id):
        return self.filenames[f_id]

    def get_object_name(self, f_id):
        return self.filenames[f_id]

    def get_class_name(self, f_id):
        return self.metadata[self.get_category(f_id)]["name"]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)


    def get_data_for_evaluation(self, idx):
        filename = self.filenames[idx]
        data_shape = np.load(os.path.join(filename, "eval","pointcloud.npz"))
        data_space = np.load(os.path.join(filename, "eval","points.npz"))
        return data_shape, data_space

    def get(self, idx):
        """Get item."""
        filename = self.filenames[idx]
        manifold_data =np.load(os.path.join(self.root, "scan", filename, str(self.scan)+".npz"))
        points_shape = manifold_data["points"]
        normals_shape = manifold_data["normals"]
        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        nls_shp = torch.tensor(normals_shape, dtype=torch.float)

        points = np.load(os.path.join(self.root, "eval", filename, "points.npz"))
        points_space = torch.tensor(points["points"], dtype=torch.float)
        occupancies = torch.tensor(np.unpackbits(points['occupancies']), dtype=torch.long)

        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=nls_shp,
                    pos_non_manifold=points_space, occupancies=occupancies, #
                    )

        return data