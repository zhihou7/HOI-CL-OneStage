### Affordance Transfer Learning for Human-Object Interaction Detection (CVPR2021), Visual Compositional Learning for Human-Object Interaction Detection (ECCV2020)

This is an implementation of ATL and VCL based on One-Stage HOI Detection.

## Getting Started


### Prerequisites

- Linux or macOS with Python ≥ 3.6
- [PyTorch](https://pytorch.org) ≥ 1.4, torchvision that matches the PyTorch installation.
- [Detectron2](https://github.com/facebookresearch/detectron2)
- Other packages listed in [reuirements.txt](./requirements.txt)

### Installation

1. Please follow the [instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2 first.
2. Install other dependencies by `pip install -r requirements.txt` or `conda install --file requirements.txt`
3. Download and prepare the data by `cd datasets; sh prepare_data.sh`.
    - The [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/) dataset and [V-COCO](https://github.com/s-gupta/v-coco) dataset.
      - If you already have, please comment out the corresponding lines in [prepare_data.sh](./prepare_data.sh) and **hard-code the dataset path using your custom path** in [lib/data/datasets/builtin.py](./lib/data/datasets/builtin.py).
    - COCO's format annotations for HICO-DET and VCOCO dataset.
    - [Glove](https://nlp.stanford.edu/projects/glove/) semantic embeddings.

## Demo Inference with Pre-trained Model

Object Detection Pretrained model (Here is the model from VCL): https://cloudstor.aarnet.edu.au/plus/s/NSkxIqfWMt9VydN

ATL model: 

## Training a model and running inference

Train Baseline (ATL)

```
python train_net_atl.py --num-gpus 2 --config-file configs/HICO-DET/interaction_R_101_FPN_pos_atl.yaml MODEL.ROI_HEADS.OBJ_IMG_NUMS 2 SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR ./output/HICO_interaction_base_101_fine1_gpu2_atl12 MODEL.ROI_HEADS.CL 0 MODEL.ROI_HEADS.CL_WEIGHT 0.25 MODEL.WEIGHTS output/model_0064999.pth
```

Train ATL

```
python train_net_atl.py --num-gpus 2 --config-file configs/HICO-DET/interaction_R_101_FPN_pos_atl.yaml MODEL.ROI_HEADS.OBJ_IMG_NUMS 2 SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR ./output/HICO_interaction_base_101_fine1_gpu2_atl12 MODEL.ROI_HEADS.CL 1 MODEL.ROI_HEADS.CL_WEIGHT 0.25 MODEL.WEIGHTS output/model_0064999.pth
```

Train VCL

```
python train_net.py --num-gpus 2 --config-file configs/HICO-DET/interaction_R_101_FPN_pos.yaml SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR ./output/HICO_interaction_base_101_fine1_gpu2_vcl MODEL.ROI_HEADS.CL 1 MODEL.ROI_HEADS.CL_WEIGHT 0.25 MODEL.WEIGHTS output/model_0064999.pth
```

Eval model
```
python train_net.py --eval-only --num-gpus 1 --config-file configs/HICO-DET/interaction_R_101_FPN_pos_atl.yaml  OUTPUT_DIR ./output/HICO_interaction_base_101_fine1_gpu2_atl12 MODEL.WEIGHTS ./output/HICO_interaction_base_101_fine1_gpu2_atl12/model_00239999.pth
```

Results

|Model|Full|Rare|Non-Rare|
|ATL |23.81 | 17.43 | 25.72|
 
### Citations
If you find this series of work are useful for you, please consider citing:

```
@inproceedings{hou2021fcl,
  title={Detecting Human-Object Interaction via Fabricated Compositional Learning},
  author={Hou, Zhi and Baosheng, Yu and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```

```
@inproceedings{hou2021vcl,
  title={Visual Compositional Learning for Human-Object Interaction Detection},
  author={Hou, Zhi and Peng, Xiaojiang and Qiao, Yu  and Tao, Dacheng},
  booktitle={ECCV},
  year={2020}
}
```

```
@inproceedings{hou2021atl,
  title={Affordance Transfer Learning for Human-Object Interaction Detection},
  author={Hou, Zhi and Baosheng, Yu and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```

### Acknowledgements

Code is built from [zero_shot_hoi](https://github.com/scwangdyd/zero_shot_hoi) and Detectron2.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
