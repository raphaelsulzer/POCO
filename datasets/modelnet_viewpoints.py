from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import glob
import logging

class ModelNetViewpointsBase(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, in_normal_field=None, **kwargs):
            
        super().__init__(root, transform, None)

        if in_normal_field is None:
            raise ValueError
        self.in_normal_field = in_normal_field

        logging.info(f"Dataset  - ShapeNet- {dataset_size}")

        version="10000"

        self.split = split
        self.filter_name = filter_name
        self.filelists = []
        self.num_non_manifold_points = num_non_manifold_points
        if split in ["train", "training"]:
            for path in glob.glob(os.path.join(self.root,f"*/convonet/{version}/train.lst")):
                self.filelists.append(path)
        elif split in ["trainVal", "trainingValidation", "training_validation"]:
            for path in glob.glob(os.path.join(self.root,f"*/convonet/{version}/train.lst")):
                self.filelists.append(path)
            for path in glob.glob(os.path.join(self.root,f"*/convonet/{version}/val.lst")):
                self.filelists.append(path)
        elif split in ["validation", "val", "test", "testing"]: # no validation set
            # for path in glob.glob(os.path.join(self.root,f"*/convonet/{version}/test.lst")):
            for path in glob.glob(os.path.join(self.root,f"*/test.lst")):
                self.filelists.append(path)
        self.filelists.sort()

        self.filenames = []

        for flist in self.filelists:
            with open(flist) as f:
                dirname = os.path.dirname(flist)
                content = f.readlines()
                content = [line.split("\n")[0] for line in content]
                content = [os.path.join(dirname,"convonet",str(version), line) for line in content]
            if dataset_size is not None:
                content = content[:dataset_size]
            self.filenames += content

        if self.filter_name is not None:
            logging.info(f"Dataset - filter {self.filter_name}")
            fname_list = []
            for fname in self.filenames:
                if self.filter_name in fname:
                    fname_list.append(fname)
            self.filenames = fname_list

        fnames = []
        for fname in self.filenames:
            if os.path.exists(fname):
                fnames.append(fname)
        self.filenames = fnames

        logging.info(f"Dataset - len {len(self.filenames)}")


        self.metadata = {}


    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-4]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.filenames[f_id].split("/")[-4]

    def get_save_dir(self, f_id):
        return "/".join(self.filenames[f_id].split("/")[-4:-2])

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
        filename = filename.replace("convonet/43", "eval")
        data_shape = np.load(os.path.join(filename, "pointcloud.npz"))
        data_space = np.load(os.path.join(filename, "points.npz"))
        return data_shape, data_space

    def get(self, idx):
        """Get item."""
        filename = self.filenames[idx]
        manifold_data =np.load(os.path.join(filename, "pointcloud.npz")) 
        points_shape = manifold_data["points"]
        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        if self.in_normal_field == "viewpoint":
            sname = "sensor_position" if("sensor_position" in manifold_data.keys()) else "sensors"
            normals_shape = manifold_data[sname] - manifold_data["points"]
        else:
            raise ValueError
            normals_shape = manifold_data["normals"]
        
        nls_shp = torch.tensor(normals_shape, dtype=torch.float)
        nls_shp = torch.nn.functional.normalize(nls_shp, dim=1)

        points = np.load(os.path.join(filename, "points.npz"))
        points_space = torch.tensor(points["points"], dtype=torch.float)
        occupancies = torch.tensor(np.unpackbits(points['occupancies']), dtype=torch.long)


        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=nls_shp,
                    pos_non_manifold=points_space, occupancies=occupancies, #
                    )

        return data


class ModelNetViewpoints(ModelNetViewpointsBase):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, in_normal_field="viewpoint", **kwargs):
            
        super().__init__(root, 
            split=split, 
            transform=transform,
            filter_name=filter_name, 
            num_non_manifold_points=num_non_manifold_points, 
            in_normal_field=in_normal_field, 
            dataset_size=dataset_size, **kwargs)