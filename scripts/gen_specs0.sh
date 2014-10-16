#!/bin/bash

# generate specs from model
# model=AE0917_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20140918_1516_37811181
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_npy/Laura_data_*' --output_dir $HOME/generated_specs/Laura_AE120_clean \
# --rectified --preprocessor GCN
# wait
#
# model=AE0919_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2214_44998436
# python $HOME/Pynet/scripts/generate_specs_from_model.py --model $HOME/Pynet/save/log/$model/cpu_model.pkl \
# --dataset '/home/smg/zhenzhou/datasets/Laura_npy/Laura_data_*' --output_dir $HOME/generated_specs/Laura_AE120_noisy \
# --rectified --preprocessor GCN
# wait



# synthesis wav from specs
# bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE120_clean --spec_ext spec.f8 \
# --wav_dir $HOME/generated_specs/Laura_AE120_clean --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
# wait
#
# bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE120_noisy --spec_ext spec.f8 \
# --wav_dir $HOME/generated_specs/Laura_AE120_noisy --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
# wait



model=AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2220_41651965
feature_size=120
python $HOME/Pynet/scripts/mgc2spec_thru_decoder_one_by_one.py \
--mgc_dir /home/smg/takaki/DNN/Zhenzhou/20140925/${feature_size} \
--mgc_ext mgc \
--mgc_txt_file $HOME/datasets/test_spec.txt \
--input_mgc_dtype f4 \
--feature_size ${feature_size} \
--output_dir $HOME/generated_specs/hmm_mgc_specs/${feature_size} \
--preprocessor GCN \
--orig_spec_dir $HOME/VCTK/data/inter-module/mcep/England/Laura \
--orig_spec_ext spec \
--orig_spec_dtype f4 \
--orig_spec_feature_size 2049 \
--output_dtype f8 \
--model $HOME/Pynet/save/log/$model/cpu_model.pkl \






# combine mgc into npy
# model=AE0917_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20140918_1516_37811181
# feature_size=120
# python $HOME/Pynet/scripts/specs2data.py \
# --spec_dir /home/smg/takaki/DNN/Zhenzhou/20140925/${feature_size} \
# --ext mgc \
# --splits 20 \
# --input_spec_dtype f4 \
# --feature_size ${feature_size} \
# --output_dir $HOME/datasets/hmm_mgc_npy/paper/${feature_size}
# wait
#
# model=AE0919_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2214_44998436
# feature_size=120
# python $HOME/Pynet/scripts/specs2data.py \
# --spec_dir /home/smg/takaki/DNN/Zhenzhou/20140925/${feature_size}_n \
# --ext mgc \
# --splits 20 \
# --input_spec_dtype f4 \
# --feature_size ${feature_size} \
# --output_dir $HOME/datasets/hmm_mgc_npy/paper/${feature_size}_n
