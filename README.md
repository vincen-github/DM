# Distribution Matching for Self-Supervised Transfer Learning
Official implementation of Distribution Matching (DM) for Self-Supervised Transfer Learning.

## Organization
The checkpoints are saved in the `models` directory every 100 epochs during training and are used to reproduce the reported results. The `dataset` directory contains code for loading datasets, the `eval` directory handles evaluation, and the `method` directory involves the implementation of DM. The `model.py` file defines the encoder structure. Creating a folder named `data` and decompressing the dataset there in advance is required.

## Supported Datasets
- CIFAR-10 
- CIFAR-100
- STL-10

## Environment
All experiments were conducted using a single Tesla V100 GPU unit. The torch version is 2.2.1+cu118 and the CUDA version is 11.8.

## Usage
```
python -u -m train --dataset cifar10 --bs 512
python -u -m train --dataset cifar100 --bs 512
python -u -m train --dataset stl10 --bs 384
```

## Citation
```
@misc{jiao2025distributionmatchingselfsupervisedtransfer,
      title={Distribution Matching for Self-Supervised Transfer Learning}, 
      author={Yuling Jiao and Wensen Ma and Defeng Sun and Hansheng Wang and Yang Wang},
      year={2025},
      eprint={2502.14424},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2502.14424}, 
}
```
