
import cPickle
import os
import glob
import argparse
import numpy as np

thisfile_path = os.path.realpath(__file__)
thisfile_dir = os.path.dirname(thisfile_path)
NNdir = os.path.dirname(thisfile_dir)
best_model = 'AE15Double_GCN_20140415_1404_50336696'





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Pipeline for combining f4 specs into \
                                                    splits to passing splits through model \
                                                    to splitting splits into f8 specs again ''')
        
    parser.add_argument('--model_name', help='''directory for model''')
    parser.add_argument('--npy_files', help='''npy files''')
    parser.add_argument('--filename_files', help='the filename files containing information about the ' +
                                                'spec files and the num of examples for the spec files')
    parser.add_argument('--save_format', default='f8', help='''output save formats f4 | f8''')

    args = parser.parse_args()
    
    print 'loading model ' + args.model_name
    with open(NNdir + '/save/log/' + args.model_name + '/model.pkl', 'rb') as pkl:
        model = cPickle.load(pkl)
    
    files = glob.glob(args.npy_files)
    assert len(files) > 0, 'empty file list'
        
    for f in files:
        print 'opening ' + f
        obj = open(f, 'rb')
        data = np.load(obj)
        assert data.shape[0] % 2049 == 0, 'data shape % 2049 is not 0'
        reshaped = data.reshape(data.shape[0]/2049, 2049)
        print '..fproping '
        out = model.fprop(reshaped)
        out = out.reshape(data.shape[0])
        assert out.shape[0] % 2049 == 0, 'output from fprop % 2049 is not 0'
        
        print 'saving fprop output %s.out'%f
        outfile = open(f + '.out', 'wb')
        np.save(outfile, out)
        
        obj.close()
        outfile.close()
        
    outfiles = args.npy_files + '.out'
    print 'passing files into generate_specs.py'
    os.system('%s/generate_specs.py --data_files=%s --filename_files=%s dtype=%s'
                %(thisfile_dir, outfiles, args.filename_files, args.save_format))
            
            
    
    
    