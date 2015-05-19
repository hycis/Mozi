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



# model=AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2220_41651965
# feature_size=120
# python $HOME/Pynet/scripts/mgc2spec_thru_decoder_one_by_one.py \
# --mgc_dir /home/smg/takaki/DNN/Zhenzhou/20140925/${feature_size} \
# --mgc_ext mgc \
# --mgc_txt_file $HOME/datasets/test_spec.txt \
# --input_mgc_dtype f4 \
# --feature_size ${feature_size} \
# --output_dir $HOME/generated_specs/hmm_mgc_specs/${feature_size} \
# --preprocessor GCN \
# --orig_spec_dir $HOME/VCTK/data/inter-module/mcep/England/Laura \
# --orig_spec_ext spec \
# --orig_spec_dtype f4 \
# --orig_spec_feature_size 2049 \
# --output_dtype f8 \
# --model $HOME/Pynet/save/log/$model/cpu_model.pkl \






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


# model1=AE1110_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_gaussian_20141111_2155_45180040
# model2=AE1111_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_blackout_continue_20141112_0842_01925797
# model3=AE1110_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_clean_20141111_2157_47387660
# model4=AE1111_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_batchout_continue_20141112_0844_45882544
# models=( $model1 $model2 $model3 $model4 )
#
# data1=AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_gaussian_continue_20141110_1250_49502872
# data2=AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_blackout_continue_20141110_1249_12963320
# data3=AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_clean_continue_20141110_1235_21624029
# data4=AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_batchout_continue_20141111_0957_22484008
# datasets=( $data1 $data2 $data3 $data4 )
#
# for ((i=0;i<${#models[@]};++i)); do
#   # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
#   # --gpu_model $SCRATCH/Pynet/save/log/$model/model.pkl \
#   # --cpu_model $SCRATCH/Pynet/save/log/$model/cpu_model.pkl
#   #
#   # wait
#   echo 'datasets: ' ${datasets[i]}
#   echo 'models: ' ${models[i]}
#
#   python $SCRATCH/Pynet/scripts/encode_dataset.py \
#   --model $SCRATCH/Pynet/save/log/${models[i]}/cpu_model.pkl \
#   --dataset "$SCRATCH/datasets/Laura/noisy/${datasets[i]}/Laura_warp_data_*.npy" \
#   --output_dir $SCRATCH/datasets/Laura/noisy/${models[i]}
# # --preprocessor Scale \
#   wait
# done


# model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141112_2145_06823495
# model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141122_1504_55793539
# model3=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_blackout_20141113_1953_00525622
# model4=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_blackout_20141122_1519_22442505
# model5=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_batchout_20141113_1309_24442341
# model6=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_batchout_20141122_1525_15102602

# model1=AE1130_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_gaussian_20141130_1520_18110100
# model2=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_gaussian_20141201_2249_40819430
# model3=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141204_2238_29999186
# model4=AE1203_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141203_2229_19073523
# model1=AE1214_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_maskout_20141213_1521_36841339
model1=AE1213_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_maskout_20141213_2121_34850799
models=( $model1 )
#



for ((i=0;i<${#models[@]};++i)); do
  # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
  # --gpu_model $SCRATCH/Pynet/save/log/$model/model.pkl \
  # --cpu_model $SCRATCH/Pynet/save/log/$model/cpu_model.pkl
  #
  # wait
  # echo 'datasets: ' ${datasets[i]}
  echo 'models: ' ${models[i]}

  # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
  # --gpu_model $SCRATCH/Pynet/save/log/${models[i]}/model.pkl \
  # --cpu_model $SCRATCH/Pynet/save/log/${models[i]}/cpu_model.pkl
  # wait

  python /home/smg/zhenzhou/Pynet/scripts/sync_model.py --from_to biaree nii --model ${models[i]}
  wait

  python /home/smg/zhenzhou/Pynet/scripts/data2mcep.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
  --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
  --output_dir generated_mceps/Scale_Warp/${models[i]} \
  --specnames 'datasets/Laura_warp_npy/Laura_warp_specnames_0*'
  wait

  python Pynet/scripts/generate_specs_from_model.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
  --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_*.npy' \
  --txt_file /home/smg/zhenzhou/datasets/test_spec.txt \
  --output_dir generated_specs/Scale_Warp/${models[i]}
  wait

  dir=/home/smg/zhenzhou/generated_specs/Scale_Warp/${models[i]}
  wav=/home/smg/zhenzhou/generated_specs/Scale_Warp/${models[i]}
  pynet=/home/smg/zhenzhou
  #
  # if [ ! -d $dir ]; then
  #   echo 'make wav dir' $dir
  #   mkdir $dir
  # fi
  #
  if [ ! -d $wav ]; then
    echo 'make wav dir' $wav
    mkdir $wav
  fi

  # rsync -uv hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/generated_specs/${models[i]}/111[0-9]_1.spec.warp.f8 $dir
  # wait
  bash $pynet/Pynet/scripts/unwarp.sh --warp_dir $dir --warp_ext spec.warp.f8 --unwarp_dir $dir --dtype f8
  wait
  bash $pynet/Pynet/scripts/synthesis.sh --vctk_home /home/smg/zhenzhou --spec_dir $dir --spec_ext spec.unwarp.f8 --wav_dir $wav --dtype f8
  wait

  # python Pynet/scripts/generate_specs_from_model.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
  # --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_000.npy' \
  # --output_dir generated_specs/${models[i]}

  # dir=/Volumes/Storage/generated_specs/Laura/${models[i]}
  # wav=/Volumes/Storage/generated_wavs/Laura/${models[i]}
  # pynet=/Volumes/Storage/Dropbox/CodingProjects
  #
  # if [ ! -d $dir ]; then
  #   echo 'make wav dir' $dir
  #   mkdir $dir
  # fi
  #
  # if [ ! -d $wav ]; then
  #   echo 'make wav dir' $wav
  #   mkdir $wav
  # fi

  # rsync -uv hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/generated_specs/${models[i]}/111[0-9]_1.spec.warp.f8 $dir
  # wait
  # bash $pynet/Pynet/scripts/unwarp.sh --warp_dir $dir --warp_ext spec.warp.f8 --unwarp_dir $dir --dtype f8
  # wait
  # bash $pynet/Pynet/scripts/synthesis.sh --vctk_home /Volumes/Storage --spec_dir $dir --spec_ext spec.unwarp.f8 --wav_dir $wav --dtype f8
  # wait

done





#
#
# # model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141112_2145_06823495
# # model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141122_1504_55793539
# model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_blackout_20141113_1953_00525622
# model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_blackout_20141122_1519_22442505
# model3=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_batchout_20141113_1309_24442341
# model4=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_batchout_20141122_1525_15102602
# model5=AE1130_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_gaussian_20141130_1520_18110100
# model6=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_gaussian_20141201_2249_40819430
# model7=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141204_2238_29999186
# model8=AE1203_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141203_2229_19073523
# models=( $model1 $model2 $model3 $model4 $model5 $model6 $model7 $model8 )
# #
#
#
#
# for ((i=0;i<${#models[@]};++i)); do
#   # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
#   # --gpu_model $SCRATCH/Pynet/save/log/$model/model.pkl \
#   # --cpu_model $SCRATCH/Pynet/save/log/$model/cpu_model.pkl
#   #
#   # wait
#   # echo 'datasets: ' ${datasets[i]}
#   echo 'models: ' ${models[i]}
#
#   # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
#   # --gpu_model $SCRATCH/Pynet/save/log/${models[i]}/model.pkl \
#   # --cpu_model $SCRATCH/Pynet/save/log/${models[i]}/cpu_model.pkl
#   # wait
#
#   # python /home/smg/zhenzhou/Pynet/scripts/sync_model.py --from_to helios nii --model ${models[i]}
#   # wait
#   #
#   # python /home/smg/zhenzhou/Pynet/scripts/data2mcep.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
#   # --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
#   # --output_dir generated_mceps/Scale_Warp/${models[i]} \
#   # --specnames 'datasets/Laura_warp_npy/Laura_warp_specnames_0*'
#   # wait
#
#   python Pynet/scripts/generate_specs_from_model.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
#   --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_*.npy' \
#   --output_dir generated_specs/Scale_Warp/${models[i]}
#
#   # dir=/Volumes/Storage/generated_specs/Laura/${models[i]}
#   # wav=/Volumes/Storage/generated_wavs/Laura/${models[i]}
#   # pynet=/Volumes/Storage/Dropbox/CodingProjects
#   #
#   # if [ ! -d $dir ]; then
#   #   echo 'make wav dir' $dir
#   #   mkdir $dir
#   # fi
#   #
#   # if [ ! -d $wav ]; then
#   #   echo 'make wav dir' $wav
#   #   mkdir $wav
#   # fi
#
#   # rsync -uv hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/generated_specs/${models[i]}/111[0-9]_1.spec.warp.f8 $dir
#   # wait
#   # bash $pynet/Pynet/scripts/unwarp.sh --warp_dir $dir --warp_ext spec.warp.f8 --unwarp_dir $dir --dtype f8
#   # wait
#   # bash $pynet/Pynet/scripts/synthesis.sh --vctk_home /Volumes/Storage --spec_dir $dir --spec_ext spec.unwarp.f8 --wav_dir $wav --dtype f8
#   # wait
#
# done
