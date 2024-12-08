# Modality-Balanced Learning for Multimedia Recommendation
This is the code for the ACM Multimedia 2021 Paper: [Modality-Balanced Learning for Multimedia Recommendation](https://dl.acm.org/doi/10.1145/3664647.3680626).

## Requirements
- Python 3.8
- PyTorch 1.9.1

## Dataset
Please refer to [this link](https://github.com/CRIPAC-DIG/LATTICE/blob/main/README.md#dataset-preparation) for the dataset preparation.

## Quick Start
Start training and inference as:
```bash
cd codes
# train visual teacher model
python train_unimodal.py --dataset baby  --model_name VBPR --train_type 2 --save_model 1
# train textual teacher model
python train_unimodal.py --dataset baby  --model_name VBPR --train_type 3 --save_model 1
# train multimodal student model
python train_kd.py --dataset baby  --model_name VBPR  
```

## Citation

Please cite our paper if you use the code:

```
@inproceedings{10.1145/3664647.3680626,
author = {Zhang, Jinghao and Liu, Guofan and Liu, Qiang and Wu, Shu and Wang, Liang},
title = {Modality-Balanced Learning for Multimedia Recommendation},
year = {2024},
url = {https://doi.org/10.1145/3664647.3680626},
doi = {10.1145/3664647.3680626},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {7551–7560},
numpages = {10},
}
```