import numpy as np
import glob
import os
import argparse

def generate_specs(datafiles, filename_files, dtype, outdir):
    
    assert dtype in ['f4', 'f8']
    
    datafiles = glob.glob(datafiles)
    filename_files = glob.glob(filename_files)
        
    data_paths = sorted(datafiles)
    filename_paths = sorted(filename_files)
    
    assert len(filename_paths) == len(data_paths) and len(filename_paths) > 0
    
    print 'npy Data paths: ', data_paths
    print 'specnames paths: ', filename_paths
    
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
            f_name = f_name.rstrip('.f4')
            f_name = f_name.rstrip('.f8')
            d_arr[pointer:pointer+int(num)].tofile(outdir + '/' + f_name+'.%s'%dtype, format=dtype)
            pointer += int(num)
        
        assert pointer == d_arr.shape[0], 'did not recur until the end of array'    
        
        f.close()
        d.close()
    
    print 'All files saved to ' + outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Generate spec files from splits of npy files''')
    parser.add_argument('--dataset', metavar='PATH', help='''path to data npy files''')
    parser.add_argument('--specnames', metavar='PATH', help='''path to specnames npy files''')
    parser.add_argument('--output_spec_dtype',  default='f8', help='''dtype of the generated spec files f4|f8, default=f8''')
    parser.add_argument('--output_dir', metavar='DIR', help='''output for the mcep file''')

    args = parser.parse_args()
    
    print('..output data files dtype: %s'%args.output_spec_dtype)

    generate_specs(args.dataset, args.specnames, args.output_spec_dtype, args.output_dir)
    
    
    


