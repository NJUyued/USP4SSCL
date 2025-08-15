# USP4SSCL

This repository contains the official PyTorch implementation for our paper:

> **Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-supervised Continual Learning**     
> **Authors**: ***[Yue Duan](https://njuyued.github.io/)\***, Taicai Chen\*, Lei Qi, Yinghuan Shi*     
> \*: Equal contribution

  - 🔗 **Quick links:** [[PDF](https://arxiv.org/pdf/2508.05316)/[Abs](https://arxiv.org/abs/2508.05316)-arXiv | [Poster]()]
     
  - 📰 **Latest news:**
      - Our paper has been accepted by the **International Conference on Computer Vision (ICCV) 2025** 🎉🎉.
  - 📑 **More of my works:**
      - 🆕 **[LATEST]** Interested in **Noisy Correspondence Learning in Cross-Modal Retrieval**? 👉 Check out our ACMMM'24 paper **PC2** [[PDF-arXiv](https://arxiv.org/pdf/2408.01349) | [Code](https://github.com/alipay/PC2-NoiseofWeb)].
      - Interested in **Semi-Supervised Learning in Fine-Grained Visual Classification (SS-FGVC)**? 👉 Check out our AAAI'24 paper **SoC** [[PDF-arXiv](https://arxiv.org/pdf/2312.12237) | [Code](https://github.com/NJUyued/SoC4SS-FGVC/)].
      - Interested in more scenarios of **SSL with mismatched distributions**? 👉 Check out our ICCV'23 paper **PRG** [[PDF-arXiv](https://arxiv.org/pdf/2308.08872) | [Code](https://github.com/NJUyued/PRG4SSL-MNAR)].
      - Interested in **robust SSL in the MNAR setting** with mismatched distributions? 👉 Check out our ECCV'22 paper **RDA** [[PDF-arXiv](https://arxiv.org/pdf/2208.04619v2) | [Code](https://github.com/NJUyued/RDA4RobustSSL)].
      - Interested in conventional SSL or more applications of **complementary labels in SSL**? 👉 Check out our TNNLS paper **MutexMatch** [[PDF-arXiv](https://arxiv.org/pdf/2203.14316) | [Code](https://github.com/NJUyued/MutexMatch4SSL/)].

## Introduction

Semi-supervised continual learning (SSCL) aims to incrementally learn from a data stream containing both labeled and unlabeled samples, which is crucial for reducing annotation costs while handling continuous data influx. SSCL presents a tripartite challenge: ensuring effective **Unlabeled Learning (UL)**, maintaining **Memory Stability (MS)** on old tasks, and preserving **Learning Plasticity (LP)** for new tasks. Previous works often address these aspects in isolation.

In this paper, we introduce **USP**, a divide-and-conquer framework designed to synergistically enhance all three facets of SSCL:

1.  **Feature Space Reservation (FSR)**: To bolster **Plasticity**, FSR constructs reserved feature locations for future classes by shaping the feature representations of old classes into an equiangular tight frame (ETF).
2.  **Divide-and-Conquer Pseudo-labeling (DCP)**: To improve **Unlabeled Learning**, DCP assigns reliable pseudo-labels to both high- and low-confidence unlabeled data, maximizing their utility.
3.  **Class-mean-anchored Unlabeled Distillation (CUD)**: To ensure **Stability**, CUD reuses the pseudo-labels from DCP to anchor unlabeled data to stable class means, using them for distillation to prevent catastrophic forgetting.

Comprehensive evaluations show that USP significantly outperforms prior SSCL methods, demonstrating its effectiveness in balancing the stability-plasticity-unlabeled learning trade-off.

<div align=center>

<img width="750px" src="/figures/intro.png"> 
<img width="750px" src="/figures/met.png"> 
 
</div>

## Requirements

All dependencies are listed in the `environment.yml` file. You can create and activate the Conda environment using the following commands:

```bash
conda env create -f environment.yml
conda activate usp4sscl
```

Key requirements include:

  - `python=3.10`
  - `pytorch=2.1.1`
  - `torchvision=0.16.1`
  - `numpy=1.26.2`
  - `scikit-learn=1.3.2`

## How to Train

### Important Arguments

  - `--dataset`: The dataset to be used. Choices: `cifar10`, `cifar100`, `cub`, `imagenet100`.
  - `--model`: The backbone model architecture. Choices: `resnet18`, `resnet20`, `resnet32`.
  - `--nb_cl_fg`: The number of classes in the initial training session.
  - `--nb_cl`: The number of new classes introduced in each incremental session.
  - `--k_shot`: The number of labeled samples per class for few-shot scenarios.
  - `--epochs`: The number of training epochs for the initial session.
  - `--epochs_new`: The number of training epochs for incremental sessions.
  - `--base_lr`: The learning rate for the initial session.
  - `--new_lr`: The learning rate for incremental sessions.
  - `--use_conloss`: Flag to enable the Equiangular Tight Frame (ETF) based contrastive loss.
  - `--include_unlabel`: Flag to enable the use of unlabeled data in training.
  - `--buffer_size`: The total size of the memory buffer for storing exemplars from past classes.
  - `--gpu`: The specific GPU ID to use for training.

### Training Examples

You can directly execute the provided shell scripts to replicate our experiments.

#### Train on CIFAR-100

```bash
bash run_cifar.sh
```

This script provides commands to run experiments on CIFAR-100 using both `train_semi.py` (our method, USP) and `train_semi_der.py` (a variant, USP-DER).

#### Train on CUB-200

```bash
bash run_cub.sh
```

This script executes `train_cub.py` with the prescribed settings for the CUB-200 dataset.

#### Train on ImageNet-100

```bash
bash run_imagenet.sh
```

This script contains commands to run experiments on the ImageNet-100 dataset using both `train_semi.py` and `train_semi_der.py`.

## Resuming Training and Evaluation

  - **To resume training from a checkpoint**: Use the `--resume` flag and specify the checkpoint path with `--model_path @your_checkpoint_path`.
  - **For evaluation**: Modify the corresponding `run_*.sh` script by setting `--epochs 0` and `--epochs_new 0`. Load the desired trained model using the `--resume` and `--model_path` flags.

## Citation

If you find our work useful for your research, please consider citing our paper:

```
@article{duan2025divide,
  title={Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-supervised Continual Learning},
  author={Duan, Yue and Chen, Taicai and Qi, Lei and Shi, Yinghuan},
  journal={arXiv preprint arXiv:2508.05316},
  year={2025}
}
```

or

```bibtex
@inproceedings{duan2025divide,
  title={Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-supervised Continual Learning},
  author={Duan, Yue and Chen, Taicai and Qi, Lei and Shi, Yinghuan},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

