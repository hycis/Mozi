#!/bin/bash

WARP_DIR=
WARP_EXT=
UNWARP_DIR=
files=

while [ "$1" != "" ]; do
    case $1 in
        --warp_dir )            shift
                                WARP_DIR=$1
                                ;;
        --warp_ext )            shift
                                WARP_EXT=$1
                                ;; 
        --unwarp_dir )          shift
                                UNWARP_DIR=$1
                                ;; 
        --warp_txt_file )       shift
                                files=`cat $1`
                                ;;
        -h | --help )           echo 'options'
                                echo '--warp_dir : directory for warp files'
                                echo '--warp_ext : extension of the warp files, input file type has to be f4'
                                echo '--unwarp_dir : directory for saving unwarp files'
                                echo '[--warp_txt_file] : path to the txt file that contains list files for unwarping'
                                exit
                                ;;
    esac
    shift
done


n=`echo $files | wc -w`
if [ $n == 0 ]; then
    files=`ls $WARP_DIR/*.$WARP_EXT | awk -F '[.]' '{print $1}'`
fi

echo 'number of files: ' `echo $files | wc -w`

if [ ! -d $UNWARP_DIR ]; then
    echo 'make warp dir' $UNWARP_DIR
    mkdir $UNWARP_DIR
fi

 
for f in $files; do
    f=`basename $f .$WARP_EXT`
    echo 'unwarping: ' $f
    freqt -m 2048 -M 2048 -a 0.77 -A 0.0  $WARP_DIR/$f.$WARP_EXT | sopr -EXP | x2x +fd > $UNWARP_DIR/$f.spec.unwarp.f8
#     x2x +df $WARP_DIR/$f.spec.warp.f4.f8 | freqt -m 2048 -M 2048 -a 0.77 -A 0.0 | sopr -EXP | x2x +fd > $UNWARP_DIR/$f.spec.unwarp.f8
done

echo 'saved to ' $UNWARP_DIR
echo 'done!'
