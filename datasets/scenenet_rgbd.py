import os
import logging
import torch
from torch_geometric.data import Dataset, Data
import importlib
from pathlib import Path
import numpy as np
import trimesh

class SceneNetRGBD(Dataset):

    
    def __init__(self,
                 root,
                 train=True,
                 transform=None, split="training", filter_name=None, dataset_size=None, 
                 point_density=None,
                 **kwargs):


        super().__init__(root, transform, None)

        logging.info("Dataset - SceneNetRGBD")

        self.split = split
        self.point_density = point_density

        self.filenames = [
            # "val/pointclouds/traj_0",
            "val/pointclouds/traj_1",
            "val/pointclouds/traj_2",
            "val/pointclouds/traj_3",
            "val/pointclouds/traj_4",
            "val/pointclouds/traj_5",
            "val/pointclouds/traj_6",
            "val/pointclouds/traj_7",
            "val/pointclouds/traj_8",
        ]
        self.filenames = [os.path.join(self.root, filename) for filename in self.filenames]
        self.filenames.sort()

        self.dataset_size = dataset_size
        if self.dataset_size is not None:
            self.filenames = self.filenames[:self.dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)

    def get_category(self, idx):
        return self.filenames[idx].split("/")[-2]

    def get_object_name(self, idx):
        return self.filenames[idx].split("/")[-1]
        
    def get_save_dir(self, f_id):
        return "/".join(self.filenames[f_id].split("/")[-3:-2])

    def get_class_name(self, idx):
        return "n/a"

    
    def get_data_for_evaluation(self, idx):
        raise NotImplementedError
        scene = self.filenames[idx]
        input_pointcloud = np.load(scene)
        return input_pointcloud, None


    def get(self, idx):
        """Get item."""

        # load the mesh
        scene_filename = self.filenames[idx]

        data = np.loadtxt(scene_filename+".xyz", dtype=np.float32)

        pos = data[:,:3]
    
        nls = data[:,3:] - pos

        pos = torch.tensor(pos, dtype=torch.float)
        nls = torch.tensor(nls, dtype=torch.float)
        nls = torch.nn.functional.normalize(nls, dim=1)
        pos_non_manifold = torch.zeros((1,3), dtype=torch.float)


        data = Data(shape_id=idx, x=torch.ones_like(pos),
                    normal=nls,
                    pos=pos, pos_non_manifold=pos_non_manifold
                    )

        return data