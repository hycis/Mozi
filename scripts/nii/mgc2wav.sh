#!/bin/bash

model1=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_blackout_20141113_1953_00525622
model2=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_blackout_20141122_1519_22442505
model3=AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_batchout_20141113_1309_24442341
model4=AE1121_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_batchout_20141122_1525_15102602
model5=AE1130_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_gaussian_20141130_1520_18110100
model6=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_gaussian_20141201_2249_40819430
model7=AE1201_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean_20141204_2238_29999186
model8=AE1203_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141203_2229_19073523
models=( $model1 $model2 $model3 $model4 $model5 $model6 $model7 $model8 )


data1=/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_120/HMM
data2=/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_120/DNN
data3=/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_180/HMM
data4=/home/smg/takaki/DNN/Zhenzhou/20150115/blackout_180/DNN

output_dir=/home/smg/zhenzhou/generated_specs/Scale_Warp
model_dir=/home/smg/zhenzhou/Pynet/save/log

for i in $model2,$data1 $model2,$data2 $model1,$data3 $model1,$data4; do IFS=","; set $i;
# for i in $model1,$data4; do IFS=","; set $i;
  echo $1
  echo $2

  IFS='/' read -ra DATA <<< "$2"
  echo ${DATA[7]}
  echo ${DATA[8]}

  save_dir=$output_dir/${DATA[7]}_${DATA[8]}

  if [ ! -d $save_dir ]; then
    echo 'make wav dir' $save_dir
    mkdir $save_dir
  fi


  python ~/Pynet/scripts/nii/mgc2spec_thru_decoder_one_by_one.py --mgc_dir $2 --mgc_ext mgc --mgc_txt_file /home/smg/zhenzhou/datasets/test_spec.txt \
  --output_dir $save_dir --preprocessor Scale --model $model_dir/$1/cpu_model.pkl
  wait

  bash ~/Pynet/scripts/nii/unwarp.sh --warp_dir $save_dir --warp_ext f8 --unwarp_dir $save_dir --dtype f8
  wait


  for spec in $save_dir/*.spec.unwarp.f4
  do
    base=`basename $spec .spec.unwarp.f4`

    # x2x +df $save_dir/$base.spec.unwarp.f8 > $save_dir/$base.spec.unwarp.f4

    /home/smg/takaki/SRC/straight/bin/synthesis_fft -f 48000 -spec -fftl 4096 \
    -shift 5 -sigp 1.2 -cornf 4000 -float -apfile $2/$base.ap $2/$base.f0.a \
    $save_dir/$base.spec.unwarp.f4 $save_dir/$base.wav
  done


done



  #
  #
  # parser = argparse.ArgumentParser(description='''Generate specs from hmm generated mgcs using the decoding part of Autoencoder''')
  # parser.add_argument('--mgc_dir', metavar='DIR', type=str, help='director of the mgc files')
  # parser.add_argument('--mgc_ext', metavar='EXT', type=str, help='path of the mgc files')
  # parser.add_argument('--mgc_txt_file', metavar='PATH', help='(Optional) path to the text that contains the list of mgc files to be processed')
  # parser.add_argument('--input_mgc_dtype', metavar='f4|f8', default='f4',
  # help='''dtype of the input mgc files f4|f8, default=f4''')
  # parser.add_argument('--feature_size', metavar='INT', default=2049, type=int,
  # help='''feature size in an example, default=2049''')
  # parser.add_argument('--output_dir', metavar='PATH', default='.',
  # help='''directory to save the combined data file''')
  # parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
  # # parser.add_argument('--orig_spec_dir', metavar='DIR', help='directory of the original spec files')
  # # parser.add_argument('--orig_spec_ext', metavar='EXT', help='extension of original spec files')
  # # parser.add_argument('--orig_spec_dtype', metavar='f4|f8', help='dtype of original spec files')
  # # parser.add_argument('--orig_spec_feature_size', metavar='INT', default=2049, type=int,
  # #                     help='''feature size of the orig spec, default=2049''')
  # parser.add_argument('--output_dtype', metavar='f4|f8', default='f8',
  # help='output datatype of spec file, f4|f8, default=f8')
  # parser.add_argument('--model', metavar='PATH', help='path for the model')
  # parser.add_argument('--rectified', action='store_true', help='rectified negative outputs to zero')
  #

# for ((i=0;i<${#models[@]};++i)); do
  # python $SCRATCH/Pynet/scripts/gpu_ae_to_cpu_ae.py \
  # --gpu_model $SCRATCH/Pynet/save/log/$model/model.pkl \
  # --cpu_model $SCRATCH/Pynet/save/log/$model/cpu_model.pkl
  #
  # wait
  # echo 'datasets: ' ${datasets[i]}
  # echo 'models: ' ${models[i]}

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


#   parser.add_argument('--mgc_dir', metavar='DIR', type=str, help='director of the mgc files')
#   parser.add_argument('--mgc_ext', metavar='EXT', type=str, help='path of the mgc files')
#   parser.add_argument('--mgc_txt_file', metavar='PATH', help='(Optional) path to the text that contains the list of mgc files to be processed')
#   parser.add_argument('--input_mgc_dtype', metavar='f4|f8', default='f4',
#   help='''dtype of the input mgc files f4|f8, default=f4''')
#   parser.add_argument('--feature_size', metavar='INT', default=2049, type=int,
#   help='''feature size in an example, default=2049''')
#   parser.add_argument('--output_dir', metavar='PATH', default='.',
#   help='''directory to save the combined data file''')
#   parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
#   parser.add_argument('--orig_spec_dir', metavar='DIR', help='directory of the original spec files')
#   parser.add_argument('--orig_spec_ext', metavar='EXT', help='extension of original spec files')
#   parser.add_argument('--orig_spec_dtype', metavar='f4|f8', help='dtype of original spec files')
#   parser.add_argument('--orig_spec_feature_size', metavar='INT', default=2049, type=int,
#   help='''feature size of the orig spec, default=2049''')
#   parser.add_argument('--output_dtype', metavar='f4|f8', default='f8',
#   help='output datatype of spec file, f4|f8, default=f8')
#   parser.add_argument('--model', metavar='PATH', help='path for the model')
#
#   python generate_specs_from_model.py --model ${models[i]}
#   --preprocessor Scale --mgc_dir --mgc_ext mgc --mgc_txt_file /home/smg/zhenzhou/datasets/test_spec.txt
#
# done
