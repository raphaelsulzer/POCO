
# POCO: Point Convolution for Surface Reconstruction for Point Clouds with Visibility Information

This repository contains the implementation of 
POCO: Point Convolution for Surface Reconstruction for Point Clouds with Visibility Information as
described in the paper 
[Deep Surface Reconstruction for Point Clouds with Visibility Information](https://arxiv.org/abs/2202.01810).

The code is largely based on the [original repository](https://github.com/valeoai/POCO).

# Data

The datasets used in this repository can be downloaded [here](https://github.com/raphaelsulzer/dsrv-data).

The pretrained models can be downloaded with:

`bash download_pretrained.sh`

# Reconstruction

For reconstructing e.g. the ModelNet10 dataset run

`python generate.py configs/modelnet/config`

where `config` should be replaced with
- `modelnetTR.yaml` for reconstruction from a point cloud (traditional POCO)
- `modelnetSV.yaml` for reconstruction from a point cloud augmented with sensor vectors
- `modelnetAP.yaml` for reconstruction from a point cloud augmented with sensor vectors and auxiliary points


## References

If you find the code or data in this repository useful, 
please consider citing

```bibtex
@misc{sulzer2022deep,
      title={Deep Surface Reconstruction from Point Clouds with Visibility Information}, 
      author={Raphael Sulzer and Loic Landrieu and Alexandre Boulch and Renaud Marlet and Bruno Vallet},
      year={2022},
      eprint={2202.01810},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@article{boulch2022poco,
  title={POCO: Point Convolution for Surface Reconstruction},
  author={Boulch, Alexandre and Marlet, Renaud},
  journal={arXiv preprint arXiv:2201.01831},
  year={2022}
}
```
