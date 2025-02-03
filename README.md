# SurgicalGaussianSurfels: Highly Accurate Surgical Surface Reconstruction Using Gaussian Surfels

### [Project Page](https://ericzzj1989.github.io/sgs)

_____________________________________________________
![introduction](assets/DIAGRAM_.png)

## 🏔️ Environment
Please follow the steps below to install dependencies to properly run this repo.

Note: For a smooth install, make sure cudatoolkit 11.8 is native to your device. Check with `nvcc --version`

```bash
git clone --recurse-submodules https://github.com/aloma85/SurgicalGaussianSurfels.git
cd SurgicalGaussianSurfels

# install install dependencies then activate.
conda env create -f environment.yml
conda activate SurgicalGaussianSurfels

# List conda-installed packages
conda list
# Or list pip-installed packages
pip list
```
Note: If having issues with .yml file, look in `dependencies.txt` to install packages manually

## 💿 Dataset
**EndoNeRF Dataset:**  
Visit [EndoNeRF](https://github.com/med-air/EndoNeRF) to download their dataset. We make use of frames from `pulling_soft_tissues` and `cutting_tissues_twice` in our experiments.

**StereoMIS Dataset:**  
Visit [StereoMIS](https://zenodo.org/records/7727692) to download their dataset. We make use of frames from `p2-7` and `p2-8`.

We advise user to structure dataset as such:
```
├── data
│   | EndoNeRF 
│     ├── pulling
        ├── pulling_soft_tissues
│     ├── cutting
        ├── cutting_tissues_twice
│   | StereoMIS
│     ├── intestine
│     ├── liver

```


## ⏳ Training
To train a model on the EndoNerf `pulling` dataset, run 
``` 
python train.py -s data/EndoNeRF/pulling/pulling_soft_tissues -m pull_output/pulling --config arguments/endonerf/pulling.py 
``` 

## ✏️ Rendering
Run this script to render the images.  

```
python render.py -m pull_output/pulling
```


## 🔎 Evaluation
Run this script to evaluate the model.  

```
python metrics.py -m pull_output/pulling
```

---
## 👍🏿 Acknowledgement



Source code is borrowed from  [2DG](https://github.com/hbb1/2d-gaussian-splatting),  [Surfels](https://github.com/turandai/gaussian_surfels),  [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGS](https://github.com/hustvl/4DGaussians), and [Deformable-3D-Gaussian](https://github.com/ingra14m/Deformable-3D-Gaussians/tree/main). We appreciate your fine work.


## 📜 Citation
If you find this work helpful, welcome to cite this paper. 
```
@InProceedings{Sunmola2025sgs,
  author    = {Idris Sunmola},
  title     = {SurgicalGaussianSurfels: Highly Accurate Surgical Surface Reconstruction Using Gaussian Surfels},
  booktitle = {},
  year      = {2025},
}
```
