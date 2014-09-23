
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
parser.add_argument('--output_spec_dtype',  default='f8', help='''dtype of the generated spec files f4|f8, default=f8''')
# parser.add_argument('--output_dir', metavar='DIR',
#                   help='directory to which to save the generated spec files')
parser.add_argument('--output_dir', metavar='DIR', help='''output dir for the mcep file''')
parser.add_argument('--specnames', metavar='PATH', help='''path to specnames npy files''')

args = parser.parse_args()

dtype = args.output_spec_dtype
outdir = args.output_dir
filename_files = args.specnames

assert dtype in ['f4', 'f8']

print 'opening model.. ' + args.model
with open(args.model) as m:
    model = cPickle.load(m)

datafiles = glob.glob(args.dataset)
filename_files = glob.glob(args.specnames)

data_paths = sorted(datafiles)
filename_paths = sorted(filename_files)

print 'npy Data paths: ', data_paths
print 'specnames paths: ', filename_paths


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

assert len(filename_paths) == len(data_paths) and len(filename_paths) > 0

for f_path, fp in zip(data_paths, filename_paths):

    print 'opening.. ' + f_path
    data_fin = open(f_path)
    dataset_raw = np.load(data_fin)

    if args.preprocessor:
        proc = getattr(procs, args.preprocessor)()
        print 'applying preprocessing: ' + args.preprocessor
        dataset_proc = proc.apply(dataset_raw)

    else:
        dataset_proc = dataset_raw

    del dataset_raw
    print 'encoding..'
    dataset_out = model.encode(dataset_proc)
    del dataset_proc

    print '..opening specnames %s'%os.path.basename(fp)
    spec_fin = open(fp)
    f_arr = np.load(spec_fin)

    d_arr = dataset_out.astype(dtype)
    del dataset_out
    num_exp = [int(num) for f_name, num in f_arr]
    assert sum(num_exp) == d_arr.shape[0], 'number of examples in data array is different from the known number'

    pointer = 0
    for f_name, num in f_arr:
        print 'f_name, num_exp : %s, %s'%(f_name, num)
        f_name = f_name.rstrip('.f4')
        f_name = f_name.rstrip('.f8')
        d_arr[pointer:pointer+int(num)].tofile(outdir + '/' + f_name+'.%s'%dtype, format=dtype)
        pointer += int(num)

    assert pointer == d_arr.shape[0], 'did not recur until the end of array'

    data_fin.close()
    spec_fin.close()
    del d_arr


print 'All files saved to ' + outdir


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='''Generate spec files from splits of npy files''')
#     parser.add_argument('--dataset', metavar='PATH', help='''path to data npy files''')
#     parser.add_argument('--specnames', metavar='PATH', help='''path to specnames npy files''')
#     parser.add_argument('--output_dir', metavar='DIR', help='''output for the mcep file''')
#
#     args = parser.parse_args()
#
#     print('..output data files dtype: %s'%args.output_spec_dtype)
#
#     generate_specs(args.dataset, args.specnames, , args.output_dir)
