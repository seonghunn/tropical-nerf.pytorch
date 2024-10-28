# Polyhedral Complex Derivation from Piecewise Trilinear Networks

![NeurIPS](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2402.10403-blue)](https://arxiv.org/abs/2402.10403)

This repository provides the source code used in the work of "Polyhedral Complex Derivation from Piecewise Trilinear Networks," published in _NeurIPS 2024_. The code is designed to be linear-time efficient using the predefined PyTorch library (v1.13+), making it easy to integrate into your workflows with the package of `tropical`.

### Getting Started

#### Installation

To set up the required Python packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

#### Preparation

This command will install the necessary dependencies for the project, ensuring a smooth execution of the provided code.

##### Download Meshes for SDF Learning

For the Stanford 3D Scanning repository,
```bash
wget http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz -P tropical/stanford
tar xvf tropical/stanford/bunny.tar.gz -C tropical/stanford

wget http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz -P tropical/stanford
tar xvf tropical/stanford/dragon_recon.tar.gz -C tropical/stanford

wget http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz -P tropical/stanford
tar xvf tropical/stanford/happy_recon.tar.gz -C tropical/stanford

wget http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz -P tropical/stanford
mkdir tropical/stanford/armadillo
gzip -d tropical/stanford/Armadillo.ply.gz
mv tropical/stanford/Armadillo.ply tropical/stanford/armadillo/Armadillo.ply

wget http://graphics.stanford.edu/pub/3Dscanrep/drill.tar.gz -P tropical/stanford
tar xvf tropical/stanford/drill.tar.gz -C tropical/stanford

wget http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz -P tropical/stanford
tar xvf tropical/stanford/lucy.tar.gz -C tropical/stanford
mkdir tropical/stanford/lucy
mv tropical/stanford/lucy.ply tropical/stanford/lucy
```

:warning: Please note that the `.tgz` files listed below contain subdirectories. Simply extract it into the root of the project directory.

For the tenth-reduced mesh of the Lucy [[Google Drive](https://drive.google.com/file/d/1-htwyfzfk41JLgPM3WZcih7mVVHtyzGW/view?usp=sharing)][[OneDrive](https://1drv.ms/u/s!AmNshZeifO7Dkl9ie89rrZM39Sc0?e=LPh8er)],
```bash
tar xvf lucy_res10.tgz
```

##### Download Pretrained SDF Models

:warning: Please note that the `.tgz` files listed below contain subdirectories. Simply extract it into the root of the project directory.

For the pretrained SDFs for the _Small_ networks [[Google Drive](https://drive.google.com/file/d/1-njnIbRfrXGD70FRYzzgs1druovPdFNU/view?usp=sharing)][[OneDrive](https://1drv.ms/u/s!AmNshZeifO7Dkl4OHOHaj5aRUzIE?e=MwAv7J)],
```bash
tar xvf stanford_small_models.tgz
```

For the pretrained SDFs for the _Small_ and _Large_ networks [[Google Drive](https://drive.google.com/file/d/1Vj67XxSqIDmB1HwEOmnIxWUO-6Xbv_B1/view?usp=sharing)][[OneDrive](https://1drv.ms/u/s!AmNshZeifO7DkmGp8k1Vvfb2ORo2?e=I0t64h)],
```bash
tar xvf stanford_models.tgz
```

### Usage

For the Stanford [bunny|drill|happy|dragon|armadillo|lucy]:
```bash
python -m tropical.stanford.train -d {bunny|drill|happy|dragon|armadillo|lucy} -e
```

Note that `-h` option gives you this:
```bash
usage: python -m tropical.stanford.train [-h]
                                         [-d {bunny,dragon,happy,armadillo,drill,lucy}]
                                         [-s SEED] [-c]
                                         [-m {small,medium,large}] [-e] [-f]

Polyhedral complex derivation from piecewise trilinear networks

optional arguments:
  -h, --help            show this help message and exit
  -d {bunny,dragon,happy,armadillo,drill,lucy}, --dataset {bunny,dragon,happy,armadillo,drill,lucy}
                        Stanford 3D scanning model name
  -s SEED, --seed SEED  Seed
  -c, --cache           Cache the trained SDF?
  -m {small,medium,large}, --model_size {small,medium,large}
                        Model size
  -e, --eval            Run evaluation?
  -f, --force           Force flat assumption to skip curve approximation.
```

### Visualization

The above script will generate the extracted meshes under the path of `./meshes`. Or you could check with our generations with this command [[Google Drive](https://drive.google.com/file/d/1-o_DwiWmw_QgdZ9HKOmPcSKAe2gB_GD_/view?usp=sharing)][[OneDrive](https://1drv.ms/u/s!AmNshZeifO7DkmCoBEpdwOx3V2hG?e=qMhD4f)]:
```bash
tar xvf meshes.tgz
```

The `*.ply` extension refers to the Polygon File Format or Stanford Triangle Format, which is a file format commonly used to store three-dimensional data. You can easily visualize them by simply dragging and dropping them onto the website of `https://3dviewer.net`.


### Miscellaneous

#### Underfitting of the SDF

This training code is not guarantee the convergence of training nor a reliable SDF. For the reproduction, please use the attached trained SDFs with the corresponding seeds. Although the seed produced the same result with our machine, it may differ in other circumstances.

#### Force to Assume the Planarity of Trilinear Regions

The `-f` option enforces the planarity assumption, bypassing curve approximation. Surprisingly, it is proved that approximating the curved hypersurface is unnecessary for precise edge subdivision. However, naively skipping this approximation can lead to failures in consecutive edge subdivision processes due to inaccuracies in identifying trilinear regions using the epsilon-tolerate sign vectors. To address this issue, we automatically modify the sign vectors to align with the engaging hypersurfaces at the current step, overriding the epsilon-based sign vectors (Def. 3.4) to set the indices of known hypersurfaces for the new vertices to one. Note that the `-f` option is set to `True` by default to ensure robustness with the learned SDF and to enhance inference speed by eliminating the need for curve approximation calculations.

#### Details for the Small Bunny

In the paper, the `Small` bunny has a slight variation in preprocessing. We used the `bunny.npy` file below (though this difference is negligible) and applied a scaling factor to fit the data within the range of `[-1, 1]`, which may affect the Chamfer distance. To utilize this preprocessing, you can use `bunny_npy` instead of `bunny` for `-d` option.

```
wget https://github.com/lzhnb/Primitive3D/raw/main/examples/data/bunny.npy -P tropical/stanford/models/.
```

Please note that the default Marching Cubes (MC) resolution for the pseudo-ground truth is set to 512^3 for the better accuracy. If you would like to change it to 256^3 for reproduction, please refer to `tropical/stanford/train.py#L331`.

### How to Cite

Should you find this repository beneficial for your work, please consider citing it.
```
@inproceedings{kim2024tropical,
  author = {{Kim, Jin-Hwa}},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS)},
  title = {Polyhedral Complex Derivation from Piecewise Trilinear Networks},
  year = {2024}
}
```


### License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
