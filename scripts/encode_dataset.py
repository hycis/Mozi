

import argparse
import cPickle
import glob
import os
import numpy as np
import pynet.datasets.preprocessor as procs

parser = argparse.ArgumentParser(description='pass numpy data file through an AutoEncoder,'
                                            + ' and save the encoded npy files')
parser.add_argument('--model', metavar='PATH', help='path for the model')
parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
parser.add_argument('--dataset', metavar='PATH', help='path to the numpy data file')
parser.add_argument('--output_dir', metavar='DIR',
                    help='directory to which to save the generated spec files')

args = parser.parse_args()


print 'opening model.. ' + args.model
with open(args.model) as m:
  model = cPickle.load(m)

dataset_files = glob.glob(args.dataset)
dataset_files.sort()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for f_path in dataset_files:

    print 'opening.. ' + f_path
    f = open(f_path)
    dataset_raw = np.load(f)

    if args.preprocessor:
        proc = getattr(procs, args.preprocessor)()
        print 'applying preprocessing: ' + args.preprocessor
        dataset_proc = proc.apply(dataset_raw)
        if proc.__class__.__name__ == "Scale":
            print "global_max", proc.max
            print "global_min", proc.min

    else:
        dataset_proc = dataset_raw

    del dataset_raw
    print 'encoding.. '
    dataset_out = model.encode(dataset_proc)
    del dataset_proc

    print 'saving to.. ' + args.output_dir
    name = os.path.basename(f_path)

    with open(args.output_dir + '/%s'%name, 'wb') as f:
        np.save(f, dataset_out)

    del dataset_out

    print 'closing files..'
    f.close()
    print 'Done!'
