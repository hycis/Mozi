#!/bin/bash
. /Volumes/Storage/VCTK/Research-Demo/fa-tts/STRAIGHT-TTS/local.conf.0
TMP_DIR=/Volumes/Storage/VCTK/fa-tts/STRAIGHT-TTS/tmp/England/p276
F0_OUTPUT=/Volumes/Storage/VCTK/data/inter-module/f0/England/p276
#AE_specs=/Volumes/Storage/Dropbox/CodingProjects/smartNN/data/p276/Scale_generated_specs
AE_specs=~/Desktop/
filename=p276_002.spec.warp.double
base=p276_002
synthesis_fft -f $rate -spec -fftl $fftlen -order $order -shift $shift -sigp 1.2 -cornf 4000 -bap -apfile ${TMP_DIR}/abs/${base}.bndap.double ${F0_OUTPUT}/${base}.f0 $AE_specs/${filename} $AE_specs/${base}.wav > ${TMP_DIR}/abs/${base}.log
