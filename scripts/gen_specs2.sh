#!/bin/bash


# model=AE0917_Warp_Blocks_3layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1859_49533796
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_warp_npy/Laura_warp_data_*' --output_dir $HOME/generated_specs/Laura_Warp_AE120_clean \
# --preprocessor GCN
# wait
#
# model=AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2220_41651965
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_warp_npy/Laura_warp_data_*' --output_dir $HOME/generated_specs/Laura_Warp_AE120_noisy \
# --preprocessor GCN
# wait

# bash $HOME/Pynet/scripts/unwarp.sh --warp_dir $HOME/generated_specs/Laura_Warp_AE120_clean --warp_ext spec.warp.f8 \
# --unwarp_dir $HOME/generated_specs/Laura_Warp_AE120_clean --dtype f8 --warp_txt_file $HOME/datasets/test_spec.txt
# wait
#
# bash $HOME/Pynet/scripts/unwarp.sh --warp_dir $HOME/generated_specs/Laura_Warp_AE120_noisy --warp_ext spec.warp.f8 \
# --unwarp_dir $HOME/generated_specs/Laura_Warp_AE120_noisy --dtype f8 --warp_txt_file $HOME/datasets/test_spec.txt
# wait


bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_Warp_AE120_clean --spec_ext spec.unwarp.f8 \
--wav_dir $HOME/generated_specs/Laura_Warp_AE120_clean --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait

bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_Warp_AE120_noisy --spec_ext spec.unwarp.f8 \
--wav_dir $HOME/generated_specs/Laura_Warp_AE120_noisy --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait
