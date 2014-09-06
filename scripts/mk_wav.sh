spec_dir=/home/smg/zhenzhou/demo/decoded_specs/AE0729_warp_3layers_finetune_20140729_1221_54548278
dir=/home/smg/takaki/DNN/Zhenzhou/20140805/AE-120/AE-120
for spec in $spec_dir/*.spec.unwarp.f8
do  
    echo $spec
    base=`basename $spec .spec.unwarp.f8`
    x2x +df $spec > $spec_dir/$base.spec.unwarp.f4
    /home/smg/takaki/SRC/SPTK-3.7/bin/x2x +fa $dir/$base.f0 > $spec_dir/$base.f0.a
    /home/smg/takaki/SRC/straight/bin/synthesis_fft -f 48000 -spec -fftl 4096 -shift 5 \
    -sigp 1.2 -cornf 4000 -float -apfile $dir/$base.ap $spec_dir/$base.f0.a \
    $spec_dir/$base.spec.unwarp.f4 $spec_dir/$base.wav
done