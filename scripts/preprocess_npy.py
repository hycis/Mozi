import argparse
import glob
import os
import numpy as np
import pynet.datasets.preprocessor as procs

parser = argparse.ArgumentParser(description='Preprocess numpy files and saved to directory')
parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
parser.add_argument('--dataset', metavar='PATH', help='path to the numpy data file')
parser.add_argument('--output_dir', metavar='DIR', 
                    help='directory to which to save preprocessed numpy files')


args = parser.parse_args()


dataset_files = glob.glob(args.dataset)
dataset_files.sort()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for f_path in dataset_files:

    print 'opening.. ' + f_path
    fin = open(f_path)
    dataset_raw = np.load(fin)

    if args.preprocessor:
        proc = getattr(procs, args.preprocessor)()
        print 'applying preprocessing: ' + args.preprocessor
        dataset_proc = proc.apply(dataset_raw)
    
    else:
        dataset_proc = dataset_raw

    basename = os.path.basename(f_path)
    basename = basename.rstrip('.npy')
    num = basename.split('_')[-1]
    
    dir_base = os.path.basename(args.output_dir)
    save_name = '{}/{}'.format(args.output_dir, 
                 dir_base.strip('_npy') + '_data_' + num + '.npy')
    print 'saving.. ' + save_name   
    with open(save_name, 'wb') as f:
        np.save(f, dataset_proc) 
    
    del dataset_raw
    del dataset_proc
    fin.close()

    

