#!/bin/bash

# model=AE0707_2layers_10blks_20140707_1733_33554775
# model=AE0707_2layers_10blks_20140707_2009_35589670
# model=AE15Double_GCN_20140415_1404_50336696

# biaree:P276
# python $SCRATCH/smartNN/scripts/generate_specs_from_model.py --model $HOME/smartNN/save/log/$model/model.pkl \
# --dataset $SCRATCH/dataset/p276_npy/p276_data_000.npy --output_dir $SCRATCH/dataset/generated_specs/p276/$model \
# --preprocessor GCN

# biaree:Laura
model=AE0713_Warp_500_20140714_1317_43818059;
python $SCRATCH/smartNN/scripts/generate_specs_from_model.py --model $SCRATCH/pynet/save/log/$model/model.pkl \
--dataset $SCRATCH/dataset/Laura_warp_npy/Laura_warp_data_000.npy --output_dir $SCRATCH/dataset/generated_specs/Laura/$model \
--preprocessor Scale

model=AE0713_Warp_500_20140714_1317_43818059;
mkdir /Volumes/Storage/generated_specs/Laura/$model; 
rsync -Pruv hycis@briaree.calculquebec.ca:/RQexec/hycis/dataset/generated_specs/Laura/$model/1119_1.spec.f8 \
/Volumes/Storage/generated_specs/Laura/$model



# #### guillimin:Laura
# #PBS -A gm-1r16-n04
# model=AE0708_1layers_12blks_logGCN_20140711_0532_10538414
# HYCIS=/sb/project/jvb-000-aa/zhenzhou
# python $HYCIS/smartNN/scripts/generate_specs_from_model.py --model $HYCIS/pynet/save/log/$model/model.pkl \
# --dataset $HYCIS/datasets/Laura_npy/Laura_data_000.npy --output_dir $HYCIS/datasets/generated_specs/Laura/$model \
# --preprocessor LogGCN