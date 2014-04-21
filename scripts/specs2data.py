import glob
import os
import argparse
import numpy as np

thisfile_path = os.path.realpath(__file__)
thisfile_dir = os.path.dirname(thisfile_path)
NNdir = os.path.dirname(thisfile_dir)

print 'smartNN directory %s'%NNdir

def generate_splits(size, splits):
    split = []
    assert splits > 0
    fraction_size = size / splits
    
    for i in xrange(splits):
        split.append((i+1) * fraction_size)
        
    if size % fraction_size > 0:
        split.pop(-1)
        split.append(size)    
    return split


def savenpy(folder_path, extension, splits, dtype, feature_size):
    
    assert dtype in ['f4', 'f8']
    assert os.path.exists(folder_path + '/spec_files'), folder_path + '/spec_files does not exist' 
        
    dataset = os.path.basename(folder_path)
    files = glob.glob(folder_path + '/spec_files/*%s'%extension)
    size = len(files)
    assert size > 0, 'empty folder'
    split = generate_splits(size, splits)
    print '..number of files %d'%size
    data = []
    file_names = []
    count = 0
    i = 0        
    for f in files:
        count += 1
        with open(f) as fb:
            clip = np.fromfile(fb, dtype='<%s'%dtype, count=-1)
            assert clip.shape[0] % feature_size == 0
            data.extend(clip)
            name = os.path.basename(f)
            file_names.append((name, clip.shape[0]/feature_size))
        
        print(str(count) + '/' + str(size) + ' opened: '  + name)
        
        if count >= split[i]:
            with open(folder_path + '/%s_data_%.3d.npy'%(dataset,i), 'wb') as npy:
                print('..saving %s_data_%.3d.npy'%(dataset,i))
                assert len(data)%feature_size == 0
                data = np.asarray(data).reshape(len(data)/feature_size, feature_size)
                np.save(npy, data)
    
            with open(folder_path + '/%s_specnames_%.3d.npy'%(dataset,i), 'wb') as npy:
                print('..saving %s_specnames_%.3d.npy'%(dataset,i))
                np.save(npy, file_names)
                
            data = []
            file_names = []
            i += 1
        
    print('all finished successfully')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''combine specs files inside splits of npy files''')
    parser.add_argument('--dataset', help='''dataset''')
    parser.add_argument('--ext', default='spec', help='''extension of the spec files''')
    parser.add_argument('--splits', default=1, type=int, help='''number of splits for the merged spec files''')
    parser.add_argument('--input_spec_dtype',  default='f4', help='''dtype of the input spec files f4|f8''')
    parser.add_argument('--feature_size', default=2049, help='''feature size in an example''')

    args = parser.parse_args()

    dataset_dir = NNdir + '/data/' + args.dataset

    print('..dataset directory %s'%dataset_dir)
    print('..extension %s'%args.ext)
    print('..number of splits %s'%args.splits)
    print('..output data files dtype: %s'%args.spec_dtype)
    
    savenpy(dataset_dir, args.ext, args.splits, args.input_spec_dtype, args.feature_size)
        
        
        
