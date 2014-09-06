import cPickle
import os
import glob
import argparse
import numpy as np
import smartNN.datasets.preprocessor as preproc
import argparse


parser = argparse.ArgumentParser(description='reconstruct spec file from input spec')
parser.add_argument('--model', metavar='PATH', help='directory of the model use for reconstruction')
parser.add_argument('--preprocessor', metavar='NAME', help='preprocessor for the model')
parser.add_argument('--spec_file', metavar='PATH', help='original spec file')
parser.add_argument('--input_spec_dtype',  metavar='f4|f8', default='f4', 
                    help='dtype of the input spec files f4|f8')
parser.add_argument('--output_spec_dtype', metavar='f4|f8', default='f8', 
                    help='output datatype of spec file, f4|f8')
parser.add_argument('--output_dir', metavar='PATH', default='./reconstructed_specs',
                    help='directory to which to save the generated spec files')
parser.add_argument('--feature_size', metavar='INT', type=int, default=2049,
                    help='feature size of each example')

args = parser.parse_args()
spec_files = glob.glob(args.spec_file)

print 'unpickling model.. ' + os.path.basename(args.model)
with open(args.model + '/model.pkl') as f:
    model = cPickle.load(f)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

print 'loading preprocessor.. ' + args.preprocessor
proc = getattr(preproc, args.preprocessor)()

for spec_file in spec_files:
    print 'opening.. ' + os.path.basename(spec_file)
    spec = np.fromfile(spec_file, '<%s'%args.input_spec_dtype)
    assert spec.shape[0] % args.feature_size == 0, 'length of of spec file ' + spec.shape[0] + \
                                                    'not multiple of feature size ' + args.feature_size
    spec = spec.reshape(spec.shape[0]/args.feature_size, args.feature_size)
    
    print 'applying preprocessing.. ' + args.preprocessor
    spec = proc.apply(spec)
    
    print 'fproping.. '
    spec = model.fprop(spec)
    
    print 'inverting preprocessing.. '
    spec = proc.invert(spec)
    spec = spec.astype(args.output_spec_dtype)
    
    import pdb
    pdb.set_trace()
    
    filename = os.path.basename(spec_file) + '.%s'%args.output_spec_dtype
    print 'saving..'
    spec.tofile(args.output_dir + '/' + filename, format='<%s'%args.output_spec_dtype)
    
print 'all reconstructed spec files save in ' + args.output_dir
