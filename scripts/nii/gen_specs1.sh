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


# bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE180_clean --spec_ext spec.f8 \
# --wav_dir $HOME/generated_specs/Laura_AE180_clean --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
# wait
#
# bash $HOME/Pynet/scripts/synthesis.sh --spec_dir $HOME/generated_specs/Laura_AE180_noisy --spec_ext spec.f8 \
# --wav_dir $HOME/generated_specs/Laura_AE180_noisy --spec_txt_file $HOME/datasets/test_spec.txt --dtype f8
# wait

# model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141112_2145_06823495
# model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141122_1504_55793539
model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_blackout_20141113_1953_00525622
model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_blackout_20141122_1519_22442505
model3=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_batchout_20141113_1309_24442341
model4=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_batchout_20141122_1525_15102602
model5=AE1130_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_gaussian_20141130_1520_18110100
model6=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_gaussian_20141201_2249_40819430
model7=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141204_2238_29999186
model8=AE1203_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141203_2229_19073523
models=( $model1 $model2 $model3 $model4 $model5 $model6 $model7 $model8 )
#

/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_120/HMM
/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_120/DNN
/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_180/HMM
/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_180/DNN

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

  # python /home/smg/zhenzhou/Pynet/scripts/sync_model.py --from_to helios nii --model ${models[i]}
  # wait
  #
  # python /home/smg/zhenzhou/Pynet/scripts/data2mcep.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
  # --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_0*' \
  # --output_dir generated_mceps/Scale_Warp/${models[i]} \
  # --specnames 'datasets/Laura_warp_npy/Laura_warp_specnames_0*'
  # wait

  # python Pynet/scripts/generate_specs_from_model.py --model Pynet/save/log/${models[i]}/cpu_model.pkl \
  # --preprocessor Scale --dataset 'datasets/Laura_warp_npy/Laura_warp_data_*.npy' \
  # --txt_file /home/smg/zhenzhou/datasets/test_spec.txt \
  # --output_dir generated_specs/Scale_Warp/${models[i]}

  # dir=/home/smg/zhenzhou/generated_specs/Scale_Warp/${models[i]}
  # wav=/home/smg/zhenzhou/generated_specs/Scale_Warp/${models[i]}
  # pynet=/home/smg/zhenzhou
  # #
  # # if [ ! -d $dir ]; then
  # #   echo 'make wav dir' $dir
  # #   mkdir $dir
  # # fi
  # #
  # if [ ! -d $wav ]; then
  #   echo 'make wav dir' $wav
  #   mkdir $wav
  # fi
  #
  # # rsync -uv hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/generated_specs/${models[i]}/111[0-9]_1.spec.warp.f8 $dir
  # # wait
  # bash $pynet/Pynet/scripts/unwarp.sh --warp_dir $dir --warp_ext spec.warp.f8 --unwarp_dir $dir --dtype f8
  # wait
  # bash $pynet/Pynet/scripts/synthesis.sh --vctk_home /home/smg/zhenzhou --spec_dir $dir --spec_ext spec.unwarp.f8 --wav_dir $wav --dtype f8
  # wait


  parser.add_argument('--mgc_dir', metavar='DIR', type=str, help='director of the mgc files')
  parser.add_argument('--mgc_ext', metavar='EXT', type=str, help='path of the mgc files')
  parser.add_argument('--mgc_txt_file', metavar='PATH', help='(Optional) path to the text that contains the list of mgc files to be processed')
  parser.add_argument('--input_mgc_dtype', metavar='f4|f8', default='f4',
  help='''dtype of the input mgc files f4|f8, default=f4''')
  parser.add_argument('--feature_size', metavar='INT', default=2049, type=int,
  help='''feature size in an example, default=2049''')
  parser.add_argument('--output_dir', metavar='PATH', default='.',
  help='''directory to save the combined data file''')
  parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
  parser.add_argument('--orig_spec_dir', metavar='DIR', help='directory of the original spec files')
  parser.add_argument('--orig_spec_ext', metavar='EXT', help='extension of original spec files')
  parser.add_argument('--orig_spec_dtype', metavar='f4|f8', help='dtype of original spec files')
  parser.add_argument('--orig_spec_feature_size', metavar='INT', default=2049, type=int,
  help='''feature size of the orig spec, default=2049''')
  parser.add_argument('--output_dtype', metavar='f4|f8', default='f8',
  help='output datatype of spec file, f4|f8, default=f8')
  parser.add_argument('--model', metavar='PATH', help='path for the model')

  python generate_specs_from_model.py --model ${models[i]}
  --preprocessor Scale --mgc_dir --mgc_ext mgc --mgc_txt_file /home/smg/zhenzhou/datasets/test_spec.txt

done
