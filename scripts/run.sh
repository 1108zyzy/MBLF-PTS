#!/bin/bash

############################ Few-shot classification
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

########################## zero-shot / training-free few-shot classification
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

######################### zero shot generalization to distribution shift. 
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