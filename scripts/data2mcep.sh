#! /bin/bash

model=AE0917_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20140918_1516_37811181
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_npy/Laura_data_0*' \
--output_dir generated_mceps/Laura_AE120_clean/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0919_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2214_44998436
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_npy/Laura_data_0*' \
--output_dir generated_mceps/Laura_AE120_noisy/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1009_07286035
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_npy/Laura_data_0*' \
--output_dir generated_mceps/Laura_AE180_clean/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140917_1013_42539511
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_npy/Laura_data_0*' \
--output_dir generated_mceps/Laura_AE180_noisy/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0917_Warp_Blocks_3layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1859_49533796
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
--output_dir generated_mceps/Laura_Warp_AE120_clean/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2220_41651965
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
--output_dir generated_mceps/Laura_Warp_AE120_noisy/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0917_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1721_46741756
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
--output_dir generated_mceps/Laura_Warp_AE180_clean/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait

model=AE0918_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140918_2113_17247388
python Pynet/scripts/data2mcep.py --model Pynet/save/log/$model/cpu_model.pkl \
--preprocessor GCN --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
--output_dir generated_mceps/Laura_Warp_AE180_noisy/ \
--specnames 'datasets/Laura_npy/Laura_specnames_0*'
wait
