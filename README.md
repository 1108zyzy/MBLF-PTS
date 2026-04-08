# **Multi-Bias Logit Fusion in a Pseudo-Text Space for CLIP Few-Shot Learning**

![](F:\MBLF-PTS-main\figures\framework.png)

## 🔍Overview

This repository contains the implementation of  MBLF-PTS for image classification with a pre-trained CLIP. We consider four task settings:  

![](F:\MBLF-PTS-main\figures\task.png)

* Zero-shot classification in a test-time adaptation manner
* Few-shot classification
* Training-free few-shot classification
* Out-of-distribution generalization

## 📐Prerequisites

### Hardware

This implementation is for the single-GPU configuration. All experiments can be reproduced on a GPU with more than 24GB memory (e.g., 3080Ti)!

### Environment 
The code is tested on PyTorch 1.13.1.

### Datasets 

We suggest downloading all datasets to a root directory (`${data_root}`), and renaming the directory of each dataset as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`. This would allow you to evaluate multiple datasets within the same run.     
If this is not feasible, you could evaluate different datasets separately, and change the `${data_root}` accordingly in the bash script.


For zero/few-shot classification, we consider 11 datasets:
* [ImageNet](https://image-net.org/index.php) 
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

For out-of-distribution generalization, we consider 4 datasets:

* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

## 🚀Run MBLF-PTS

We provide a simple bash script under `./scripts/run.sh`. You can modify the paths and other args in the script. One can easily reproduce all results by:    

```shell
bash ./scripts/run.sh
```

#### zero-shot / training-free few-shot classification:

```shell
data_root='/root/autodl-tmp/my/data/clip_tuning_datasets/'
testsets=DTD
num_important_channel=250/400/0
lambda_ape=0.3/0.7
lr=0.0001
epoch=20
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 0
do
  python ./mblf-pts_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 20   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}  \
  --use_bridge_branch \
  --bridge_w 0.12
done
```

#### zero shot generalization to distribution shift:

```shell
data_root='/root/autodl-tmp/my/data/clip_tuning_datasets/'
testsets=A
num_important_channel=250/400/0
lambda_ape=0.3/0.7
lr=0.0001
epoch=20
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 0
do
  python ./mblf-pts_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 20   --beta 8  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}  \
  --use_bridge_branch \
  --bridge_w 0.12
done
```

#### Few-shot classification:

```shell
data_root='/root/autodl-tmp/my/data/clip_tuning_datasets/'
testsets=DTD
num_important_channel=0
lambda_ape=0.3
lr=0.0001/0.001
epoch=20/100
arch=ViT-B/16
#arch=RN50
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 1
do
  python ./mblf-pts_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 10   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft \
  --tda --tda_pos_k 2 --tda_neg_k 1 --tda_pos_beta 5.0 --tda_neg_beta 4.5 \
  --tda_neg_entropy_low 0.3 --tda_neg_entropy_high 0.55 --tda_neg_mask_low 0.03 --tda_neg_mask_high 0.30 \
  --use_sakr_branch --sakr_beta 4.0 --sakr_lambda 1.6 \
  --use_bridge_branch \
  --bridge_w 0.18
done
```

##### More shot settings:

```shell
for nshot in 2
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 20   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft \
  --tda --tda_pos_k 3 --tda_neg_k 1 --tda_pos_beta 5.5 --tda_neg_beta 4.8 \
  --tda_neg_entropy_low 0.28 --tda_neg_entropy_high 0.55 --tda_neg_mask_low 0.03 --tda_neg_mask_high 0.30 \
  --use_proker_branch --proker_beta 5.0 --proker_lambda 1.4 \
  --use_bridge_branch \
  --bridge_w 0.22 \
  --bridge_train --bridge_epochs 20 --bridge_lr 0.001 --bridge_weight_decay 0.01 
done

for nshot in 4
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 20   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft \
  --tda --tda_pos_k 4 --tda_neg_k 2 --tda_pos_beta 6.0 --tda_neg_beta 5.0 \
  --tda_neg_entropy_low 0.25 --tda_neg_entropy_high 0.55 --tda_neg_mask_low 0.03 --tda_neg_mask_high 0.30 \
  --use_proker_branch --proker_beta 6.0 --proker_lambda 1.2 \
  --use_bridge_branch \
  --bridge_w 0.25 \
  --bridge_train --bridge_epochs 20 --bridge_lr 0.001 --bridge_weight_decay 0.01 
done

for nshot in 8
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 24   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft \
  --tda --tda_pos_k 7 --tda_neg_k 3 --tda_pos_beta 6.5 --tda_neg_beta 5.2 \
  --tda_neg_entropy_low 0.25 --tda_neg_entropy_high 0.55 --tda_neg_mask_low 0.03 --tda_neg_mask_high 0.30 \
  --use_proker_branch --proker_beta 7.5 --proker_lambda 1.0 \
  --use_bridge_branch \
  --bridge_w 0.28 \
  --bridge_train --bridge_epochs 20 --bridge_lr 0.001 --bridge_weight_decay 0.01 
done

for nshot in 16
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_searched_vit  \
  --gpu 0 --n_shot ${nshot} --n_augview 64   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft \
  --tda --tda_pos_k 8 --tda_neg_k 3 --tda_pos_beta 7.0 --tda_neg_beta 5.5 \
  --tda_neg_entropy_low 0.22 --tda_neg_entropy_high 0.55 --tda_neg_mask_low 0.03 --tda_neg_mask_high 0.30 \
  --use_proker_branch --proker_beta 8.0 --proker_lambda 0.9 \
  --use_bridge_branch \
  --bridge_w 0.30 \
  --bridge_train --bridge_epochs 20 --bridge_lr 0.001 --bridge_weight_decay 0.01 
done
```

## 📜Main Results

#### Zero-shot Classification
![zs](F:\MBLF-PTS-main\figures\zs.png)


#### Few-shot Classification

![fs](F:\MBLF-PTS-main\figures\fs.png)


#### Out-of-Distribution Generalization

![ood](F:\MBLF-PTS-main\figures\ood.png)

#### **I2T-w Visualization**

<img src="F:\MBLF-PTS-main\figures\umap.png" style="zoom:67%;" />

![](F:\MBLF-PTS-main\figures\reli.png)
