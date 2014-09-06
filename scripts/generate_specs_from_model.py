import argparse
import cPickle
import glob
import os
import numpy as np
import pynet.datasets.preprocessor as procs

parser = argparse.ArgumentParser(description='pass numpy data file through an AutoEncoder,'
                                            + ' and generate spec files from the AE outputs.')
parser.add_argument('--model', metavar='PATH', help='path for the model')
parser.add_argument('--preprocessor', metavar='NAME', nargs='+',help='name of the preprocessor')
parser.add_argument('--dataset', metavar='PATH', help='path to the numpy data file')
parser.add_argument('--output_dir', metavar='DIR', 
                    help='directory to which to save the generated spec files')
parser.add_argument('--output_dtype', metavar='f4|f8', default='f8', 
                    help='output datatype of spec file, f4|f8, default=f8')

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
        for processor in args.preprocessor:
            proc = getattr(procs, processor)()
            print 'applying preprocessing: ' + processor
            dataset_proc = proc.apply(dataset_raw)
    
    else:
        dataset_proc = dataset_raw

    del dataset_raw
    print 'forward propagation..'
    dataset_out = model.fprop(dataset_proc)
    del dataset_proc
    
    if args.preprocessor:
        args.preprocessor.reverse()
        for processor in args.preprocessor:
            print 'invert dataset: ' + processor
            dataset = proc.invert(dataset_out)
    else:
        dataset = dataset_out
    
    dataset = dataset.astype(args.output_dtype)
    del dataset_out

    name = os.path.basename(f_path)
    name = name.replace('data', 'specnames')
    
    print 'opening.. ' + name
    g = open(os.path.dirname(f_path) + '/' + name)
    
    names_arr = np.load(g)
    
    num_exp = [int(num) for f_name, num in names_arr]
    assert sum(num_exp) == dataset.shape[0], 'number of examples in data array is different from the spec files'
     
    pointer = 0
    for f_name, num in names_arr:
        print 'f_name, num_exp : %s, %s'%(f_name, num)
        f_name = f_name.rstrip('.f4')
        f_name = f_name.rstrip('.f8')
        dataset[pointer:pointer+int(num)].tofile(args.output_dir + '/' + f_name+'.%s'%args.output_dtype, format=args.output_dtype)
        pointer += int(num)
    
    assert pointer == dataset.shape[0], 'did not recur until the end of array'    
    
    print 'closing files..'
    f.close()
    g.close()
    print 'Done!'
    





