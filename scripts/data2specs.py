import numpy as np
import glob
import os
import argparse

thisfile_path = os.path.realpath(__file__)
thisfile_dir = os.path.dirname(thisfile_path)
NNdir = os.path.dirname(thisfile_dir)

print 'smartNN directory %s'%NNdir

def generate_specs(datafiles, filename_files, dtype):
    
    assert dtype in ['f4', 'f8']
    
    datafiles = glob.glob(datafiles)
    filename_files = glob.glob(filename_files)
        
    data_paths = sorted(datafiles)
    filename_paths = sorted(filename_files)
    
    assert len(filename_paths) == len(data_paths) and len(filename_paths) > 0
    
    print 'npy Data paths: ', data_paths
    print 'specnames paths: ', filename_paths
    
    outdir = os.path.dirname(datafiles[0]) + '/generated_specs'
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    for fp, dp in zip(filename_paths, data_paths):
        
        print '..opening %s'%os.path.basename(fp)
        print '..opening %s'%os.path.basename(dp)

        f = open(fp)
        d = open(dp)
        
        f_arr = np.load(f)
        d_arr = np.load(d)
        d_arr = d_arr.astype(dtype)
        num_exp = [int(num) for f_name, num in f_arr]
        assert sum(num_exp) == d_arr.shape[0], 'number of examples in data array is different from the known number'
        
        pointer = 0
        for f_name, num in f_arr:
            print 'f_name, num_exp : %s, %s'%(f_name, num)
            d_arr[pointer:pointer+int(num)].tofile(outdir + '/' + f_name+'.%s'%dtype, format=dtype)
            pointer += int(num)
        
        assert pointer == d_arr.shape[0], 'did not recur until the end of array'    
        
        f.close()
        d.close()
    
    print 'All files saved to ' + outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Generate spec files from splits of npy files''')
    parser.add_argument('--dataset', help='''name of dataset''')
    parser.add_argument('--ext', default='out', help='''extension of the npy files''')
    parser.add_argument('--output_spec_dtype',  default='f8', help='''dtype of the generated spec files f4|f8''')


    args = parser.parse_args()
    
    dataset_dir = NNdir + '/data/' + args.dataset
    
    
    print('..dataset directory: %s'%dataset_dir)
    print('..extension: %s'%args.ext)
    print('..output data files dtype: %s'%args.output_spec_dtype)
    
    
    filenames = dataset_dir + '/%s_specnames_???.npy'%args.dataset
    data_files = dataset_dir + '/%s_data_???.npy.%s'%(args.dataset, args.ext)
    
    generate_specs(data_files, filenames, args.output_spec_dtype)
    
    
    


