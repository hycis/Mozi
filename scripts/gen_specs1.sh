#!/bin/bash


# model=AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1009_07286035
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_npy/Laura_data_*' --output_dir $HOME/generated_specs/Laura_AE180_clean \
# --rectified --preprocessor GCN
# wait
#
# model=AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140917_1013_42539511
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_npy/Laura_data_*' --output_dir $HOME/generated_specs/Laura_AE180_noisy \
# --rectified --preprocessor GCN
# wait


bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE180_clean --spec_ext spec.f8 \
--wav_dir $HOME/generated_specs/Laura_AE180_clean --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait

bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE180_noisy --spec_ext spec.f8 \
--wav_dir $HOME/generated_specs/Laura_AE180_noisy --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
wait
