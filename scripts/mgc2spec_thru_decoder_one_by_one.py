import glob
import cPickle
from pynet.datasets.preprocessor import GCN
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='''Generate specs from hmm generated mgcs using the decoding part of Autoencoder''')
parser.add_argument('--mgc_dir', metavar='DIR', type=str, help='director of the mgc files')
parser.add_argument('--mgc_ext', metavar='EXT', type=str, help='path of the mgc files')
parser.add_argument('--mgc_txt_file', metavar='PATH', help='(Optional) path to the text that contains the list of mgc files to be processed')
parser.add_argument('--input_mgc_dtype', metavar='f4|f8', default='f4',
                    help='''dtype of the input mgc files f4|f8, default=f4''')
parser.add_argument('--feature_size', metavar='INT', default=2049, type=int,
                    help='''feature size in an example, default=2049''')
parser.add_argument('--output_dir', metavar='PATH', default='.',
                    help='''directory to save the combined data file''')
parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
parser.add_argument('--orig_spec_dir', metavar='DIR', help='directory of the original spec files')
parser.add_argument('--orig_spec_ext', metavar='EXT', help='extension of original spec files')
parser.add_argument('--orig_spec_dtype', metavar='f4|f8', help='dtype of original spec files')
parser.add_argument('--orig_spec_feature_size', metavar='INT', default=2049, type=int,
                    help='''feature size of the orig spec, default=2049''')
parser.add_argument('--output_dtype', metavar='f4|f8', default='f8',
                help='output datatype of spec file, f4|f8, default=f8')
parser.add_argument('--model', metavar='PATH', help='path for the model')
parser.add_argument('--rectified', action='store_true', help='rectified negative outputs to zero')


args = parser.parse_args()

print('..mgc dir: %s'%args.mgc_dir)
print('..mgc ext: %s'%args.mgc_ext)
print('..mgc text file: %s'%args.mgc_txt_file)
print('..input mgc dtype: %s'%args.input_mgc_dtype)
print('..feature_size: %s'%args.feature_size)
print('..save outputs to: %s'%args.output_dir)
print('..preprocessor: %s'%args.preprocessor)
print('..orig_spec_dir: %s'%args.orig_spec_dir)
print('..orig_spec_ext: %s'%args.orig_spec_ext)
print('..orig_spec_dtype: %s'%args.orig_spec_dtype)
print('..orig_spec_feature_size: %s'%args.orig_spec_feature_size)
print('..output_dtype: %s'%args.output_dtype)
print('..model: %s'%args.model)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if args.mgc_txt_file is None:
    mgc_files = glob.glob('%s/%s'%(args.mgc_dir, args.mgc_ext))
else:
    mgc_files = []
    with open(args.mgc_txt_file) as fin:
        for line in fin:
            line = line.strip()
            mgc_files.append('%s/%s.%s'%(args.mgc_dir, line, args.mgc_ext))

print '..number of mgc files: ', len(mgc_files)
print '..load model: ', args.model
with open(args.model) as md_fin:
    model = cPickle.load(md_fin)

# gcn = GCN()
mlp='/home/smg/zhenzhou/Pynet/save/log/AE0928_Warp_Laura_Blocks_GCN_Mapping_20140928_2252_22849886/cpu_model.pkl'
fin = open(mlp)
print 'load mlp..'
mlp = cPickle.load(fin)
fin.close()

for mgc in mgc_files:
    f_name = mgc.split('/')[-1]
    basename = f_name.split('.')[0]
    print 'processing: ', basename
    specfile = '%s/%s.%s'%(args.orig_spec_dir, basename, args.orig_spec_ext)

    with open(mgc) as fb, open(specfile) as spec_fin:
        print '..opening orig spec: ', specfile
        orig_spec = np.fromfile(spec_fin, dtype='<%s'%args.orig_spec_dtype, count=-1)
        assert orig_spec.shape[0] % args.orig_spec_feature_size == 0, \
            'orig_spec.shape[0]:%s, feature_size:%s'%(orig_spec.shape[0], args.orig_spec_feature_size)
        spec_npy = orig_spec.reshape((orig_spec.shape[0]/args.orig_spec_feature_size, args.orig_spec_feature_size))

        print '..opening mgc: ', mgc
        mgc_arr = np.fromfile(fb, dtype='<%s'%args.input_mgc_dtype, count=-1)
        assert mgc_arr.shape[0] % args.feature_size == 0, \
            'clip.shape[0]:%s, feature_size:%s'%(mgc_arr.shape[0], args.feature_size)
        mgc_npy = mgc_arr.reshape((mgc_arr.shape[0]/args.feature_size, args.feature_size))

        # if args.preprocessor:
        #     print '..applyin preprocessing: ', gcn.__class__.__name__
        #     gcn.apply(spec_npy)

        print '..decoding mgc'
        output = model.decode(mgc_npy)
        # import pdb
        # pdb.set_trace()
        print '..inverting decoded mgc'
        # output = gcn.invert(output)
        output = output * mlp.fprop(output)

        if args.rectified:
            print '..rectifying negatives to zero'
            output = output - (output < 0) * output

        spec = output.reshape((output.shape[0] * output.shape[1],))
        spec = spec.astype(args.output_dtype)

        spec.tofile(args.output_dir + '/' + basename+'.%s'%args.output_dtype, format=args.output_dtype)
