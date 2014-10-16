#!/bin/bash


# model=AE0917_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1721_46741756
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_warp_npy/Laura_warp_data_*' --output_dir $HOME/generated_specs/Laura_Warp_AE180_clean \
# --preprocessor GCN
# wait
#
# model=AE0918_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140918_2113_17247388
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_warp_npy/Laura_warp_data_*' --output_dir $HOME/generated_specs/Laura_Warp_AE180_noisy \
# --preprocessor GCN
# wait

# bash $HOME/Pynet/scripts/unwarp.sh --warp_dir $HOME/generated_specs/Laura_Warp_AE180_clean --warp_ext spec.warp.f8 \
# --unwarp_dir $HOME/generated_specs/Laura_Warp_AE180_clean --dtype f8 --warp_txt_file $HOME/datasets/test_spec.txt
# wait
#
# bash $HOME/Pynet/scripts/unwarp.sh --warp_dir $HOME/generated_specs/Laura_Warp_AE180_noisy --warp_ext spec.warp.f8 \
# --unwarp_dir $HOME/generated_specs/Laura_Warp_AE180_noisy --dtype f8 --warp_txt_file $HOME/datasets/test_spec.txt
# wait

bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_Warp_AE180_clean --spec_ext spec.unwarp.f8 \
--wav_dir $HOME/generated_specs/Laura_Warp_AE180_clean --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait

bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_Warp_AE180_noisy --spec_ext spec.unwarp.f8 \
--wav_dir $HOME/generated_specs/Laura_Warp_AE180_noisy --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait
