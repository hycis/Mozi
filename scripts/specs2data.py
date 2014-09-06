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


def savenpy(spec_files, splits, dtype, feature_size, output_dir):

    assert dtype in ['f4', 'f8']

    # dataset = os.path.basename(os.path.dirname(spec_files))
    files = glob.glob(spec_files)
    files.sort()
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
            assert clip.shape[0] % feature_size == 0, \
                'clip.shape[0]:%s, feature_size:%s'%(clip.shape[0],feature_size)
            data.extend(clip)
            name = os.path.basename(f)
            file_names.append((name, clip.shape[0]/feature_size))

        print(str(count) + '/' + str(size) + ' opened: '  + name)

        if count >= split[i]:
            with open(output_dir + '/data_%.3d.npy'%i, 'wb') as npy:
                print('..saving data_%.3d.npy'%i)
                assert len(data)%feature_size == 0
                data = np.asarray(data).reshape(len(data)/feature_size, feature_size)
                np.save(npy, data)

            with open(output_dir + '/specnames_%.3d.npy'%i, 'wb') as npy:
                print('..saving specnames_%.3d.npy'%i)
                np.save(npy, file_names)

            data = []
            file_names = []
            i += 1

    print('all files saved to %s'%output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''combine specs files inside splits of npy files''')
    parser.add_argument('--spec_dir', metavar='DIR', type=str, help='''dir of the spec files''')
    parser.add_argument('--ext', metavar='EXT', default='spec', help='''extension of spec files''')
    parser.add_argument('--splits', metavar='INT', default=1, type=int,
                        help='''number of splits for the merged spec files''')
    parser.add_argument('--input_spec_dtype', metavar='f4|f8', default='f4',
                        help='''dtype of the input spec files f4|f8, default=f4''')
    parser.add_argument('--feature_size', metavar='INT', default=2049, type=int,
                        help='''feature size in an example''')
    parser.add_argument('--output_dir', metavar='PATH', default='.',
                        help='''directory to save the combined data file''')

    args = parser.parse_args()

    print('..dataset directory: %s'%args.spec_dir)
    print('..spec extension: %s'%args.ext)
    print('..number of splits: %s'%args.splits)
    print('..input data files dtype: %s'%args.input_spec_dtype)
    print('..feature_size: %s'%args.feature_size)
    print('..save outputs to: %s'%args.output_dir)

    spec_files = "%s/*.%s"%(args.spec_dir, args.ext)
    print(spec_files)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    savenpy(spec_files, args.splits, args.input_spec_dtype, args.feature_size, args.output_dir)
