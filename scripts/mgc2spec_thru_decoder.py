import glob
import cPickle
import os
import argparse
import numpy as np
import pynet.datasets.preprocessor as procs

def generate_specs(ext, spec_dir, specnames, datafiles, dtype, feature_size, 
                   output_dir, preprocessor, output_dtype, model):
    
    assert dtype in ['f4', 'f8']
    
    print 'opening model.. ' + model
    with open(model) as m:
        model = cPickle.load(m)
    
    specnames_ls = glob.glob(specnames)
    datafiles_ls = glob.glob(datafiles)
    specnames_ls.sort()
    datafiles_ls.sort()
    
    spec_files = "%s/*.%s"%(spec_dir, ext)
    files = glob.glob(spec_files)
    size = len(files)
    assert size > 0, 'empty mgc folder'
    print '..number of mgc files %d'%size
    data = []
    count = 0
    
    for datafile, specname in zip(datafiles_ls, specnames_ls):
        print 'datafile: ' + datafile
        print 'specname: ' + specname
        assert datafile.split('_data_')[-1] == specname.split('_specnames_')[-1]
        
        specname_fin = open(specname)
        specname_data = np.load(specname_fin)
        
        data = []
        mgc_frame_num = []
        for name, num_frames in specname_data:
            basename = name.split('.')[0]
            f = '%s/%s.%s'%(spec_dir, basename, ext)
            print '..opening ' + f            
            count += 1

            clip = np.fromfile(f, dtype='<%s'%dtype, count=-1)
            assert clip.shape[0] % feature_size == 0, \
                  'clip.shape[0]:%s, feature_size:%s'%(clip.shape[0],feature_size)
            
            mgc_frame_num.append(clip.shape[0] / feature_size)
            print '(mgc frame num, spec frame num)', clip.shape[0] / feature_size, int(num_frames)
            data.extend(clip)
                
            print(str(count) + '/' + str(size) + ' opened: '  + name)

        specname_basename = os.path.basename(specname)
        data_basename = specname_basename.replace('specnames', 'data')
        assert len(data) % feature_size == 0
        print '..reshaping mgc files into npy array' 
        low_dim_data = np.asarray(data).reshape(len(data)/feature_size, feature_size)

        data_fin = open(datafile)
        dataset_raw = np.load(data_fin)

        if preprocessor:
            proc = getattr(procs, args.preprocessor)()
            print 'applying preprocessing: ' + args.preprocessor
            proc.apply(dataset_raw)
        
        del dataset_raw
        
        print 'decoding..'
        dataset_out = model.decode(low_dim_data)
        del low_dim_data
        
        if preprocessor:
            print 'invert dataset..'
            dataset = proc.invert(dataset_out)
        else:
            dataset = dataset_out
            
        dataset = dataset.astype(output_dtype)
        del dataset_out

        pointer = 0
        for specname_d, mgc_num in zip(specname_data, mgc_frame_num):
            f_name, num = tuple(specname_d)
            print 'f_name, mgc_num_frames : %s, %s'%(f_name, mgc_num)
            f_name = f_name.rstrip('.f4')
            f_name = f_name.rstrip('.f8')
            dataset[pointer:pointer + mgc_num].tofile(output_dir + '/' + f_name+'.%s'%output_dtype, format=output_dtype)
            pointer += mgc_num

        assert pointer == dataset.shape[0], 'did not recur until the end of array'    

        print 'closing files..'
        data_fin.close()
        specname_fin.close()
        print 'Done!'

        
    print('all files saved to %s'%output_dir)
    
   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Generate specs from hmm generated mgcs using the decoding part of Autoencoder''')
    parser.add_argument('--mgc_dir', metavar='DIR', type=str, help='''dir of the mgc files''')
    parser.add_argument('--ext', metavar='EXT', default='spec', help='''extension of mgc files''')
    parser.add_argument('--specnames', metavar='PATH', help='''path to specnames npy files''')
    parser.add_argument('--dataset', metavar='PATH', help='path to data npy file')
    parser.add_argument('--input_spec_dtype', metavar='f4|f8', default='f4', 
                        help='''dtype of the input spec files f4|f8, default=f4''')
    parser.add_argument('--feature_size', metavar='INT', default=2049, type=int, 
                        help='''feature size in an example, default=2049''')
    parser.add_argument('--output_dir', metavar='PATH', default='.', 
                        help='''directory to save the combined data file''')
    parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
    parser.add_argument('--output_dtype', metavar='f4|f8', default='f8', 
                    help='output datatype of spec file, f4|f8, default=f8')
    parser.add_argument('--model', metavar='PATH', help='path for the model')


    args = parser.parse_args()

    print('..dataset directory: %s'%args.mgc_dir)
    print('..spec extension: %s'%args.ext)
    print('..specnames: %s'%args.specnames)
    print('..original npy dataset: %s'%args.dataset)
    print('..input data files dtype: %s'%args.input_spec_dtype)
    print('..feature_size: %s'%args.feature_size)
    print('..save outputs to: %s'%args.output_dir)
    print('..preprocessor: %s'%args.preprocessor)
    print('..output_dtype: %s'%args.output_dtype)
    print('..model: %s'%args.model)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    generate_specs(args.ext, args.mgc_dir, args.specnames, args.dataset, args.input_spec_dtype, 
            args.feature_size, args.output_dir, args.preprocessor, args.output_dtype, args.model)
        
        
        
