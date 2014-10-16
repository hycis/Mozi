#!/bin/bash

VCTK_HOME=
SPEC_DIR=
SPEC_EXT=
WAV_DIR=
files=
dtype=

while [ "$1" != "" ]; do
    case $1 in
        --vctk_home)            shift
                                VCTK_HOME=$1
                                ;;
        --spec_dir )            shift
                                SPEC_DIR=$1
                                ;;
        --spec_ext )            shift
                                SPEC_EXT=$1
                                ;;
        --wav_dir )             shift
                                WAV_DIR=$1
                                ;;
        --spec_txt_file )       shift
                                files=`cat $1`
                                ;;
        --dtype )               shift
                                dtype=$1
                                ;;
        -h | --help )           echo 'options'
                                echo '--vctk_home : home directory of the VCTK folder'
                                echo '--spec_dir : directory for spec files'
                                echo '--spec_ext : extension of the spec files, input file type has to be f8'
                                echo '--wav_dir : directory for saving the output wav files'
                                echo '[--spec_txt_file] : path to the txt file that contains list of files for processing'
                                echo '--dtype : dtype of input files, f4|f8'
                                exit
                                ;;
    esac
    shift
done


# VCTK_HOME=/Volumes/Storage
VCTK_HOME=/home/smg/zhenzhou

. $VCTK_HOME/VCTK/Research-Demo/fa-tts/STRAIGHT-TTS/local.conf.0
TMP_DIR=$VCTK_HOME/VCTK/Research-Demo/fa-tts/STRAIGHT-TTS/tmp/England/Laura
F0_OUTPUT=$VCTK_HOME/VCTK/data/inter-module/f0/England/Laura


n=`echo $files | wc -w`
if [ $n == 0 ]; then
    files=`ls $SPEC_DIR/*.$SPEC_EXT | awk -F '[.]' '{print $1}'`
fi


echo 'number of files: ' `echo $files | wc -w`
# echo $files

if [ ! -d $WAV_DIR ]; then
    echo 'make wav dir' $WAV_DIR
    mkdir $WAV_DIR
fi


for f in $files; do
    base=`basename $f .$SPEC_EXT`
    filename="$base.$SPEC_EXT";
    echo '----------------'
    echo $base
    echo 'spec file: ' $SPEC_DIR/${filename};
    if [ $dtype == 'f8' ]; then
        synthesis_fft -f $rate -spec -fftl $fftlen -order $order -shift $shift -sigp 1.2 \
        -cornf 4000 -bap -apfile ${TMP_DIR}/abs/${base}.bndap.double ${F0_OUTPUT}/${base}.f0 \
        $SPEC_DIR/${filename} $WAV_DIR/${base}.wav > ${TMP_DIR}/log/${base}.log;
    elif [ $dtype == 'f4' ]; then
        # echo 'converting f8 to f4, save to: ' $SPEC_DIR/$base.f4
        # x2x +fd $SPEC_DIR/${filename} > $SPEC_DIR/$base.f4
        # echo 'synthesising from: ' $SPEC_DIR/$base.f4
        synthesis_fft -float -f $rate -spec -fftl $fftlen -order $order -shift $shift -sigp 1.2 \
        -cornf 4000 -bap -apfile ${TMP_DIR}/abs/${base}.bndap.double ${F0_OUTPUT}/${base}.f0 \
        $SPEC_DIR/${filename} $WAV_DIR/${base}.wav > ${TMP_DIR}/log/${base}.log;
    else
        echo 'error: dtype not f4 | f8'
    fi
done
