

import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(description='''get the specfiles for training validating and testing''')
parser.add_argument('--specname_dir', metavar='DIR', required=True, help='the path to the specname npy file')
parser.add_argument('--train_valid_test_ratio', metavar='NUMBER STRING', required=True, help='''format '1 2 3' ''')
parser.add_argument('--save', help='save to text')
args = parser.parse_args()

name_dir = args.specname_dir
spec_files = glob.glob(name_dir + '/Laura_warp_specnames_*.npy')

tvt_ratio = args.train_valid_test_ratio.strip().split(' ')
tvt_ratio = [int(s) for s in tvt_ratio]

assert len(tvt_ratio) == 3

if args.save:
    train_spec = open('train_spec.txt', 'wb')
    valid_spec = open('valid_spec.txt', 'wb')
    test_spec = open('test_spec.txt', 'wb')

for name_path in spec_files:
    fin = open(name_path)
    data = np.load(fin)

    ttl_frames = 0
    for pair in data:
        ttl_frames += int(pair[1])

    train_valid_break = tvt_ratio[0] * 1.0 / sum(tvt_ratio) * ttl_frames
    valid_test_break = (tvt_ratio[0] + tvt_ratio[1])  * 1.0 / sum(tvt_ratio) * ttl_frames

    ttl_frames = 0
    count = 0
    for pair in data:

        if ttl_frames < train_valid_break:
            print 'train: %s : %s'%tuple(pair)
            if args.save:
                train_spec.write(pair[0].split('.')[0] + '\n')
            count += 1

        elif ttl_frames < valid_test_break:
            print 'valid: %s : %s'%tuple(pair)
            if args.save:
                valid_spec.write(pair[0].split('.')[0] + '\n')
            count += 1
        else:
            print 'test: %s : %s'%tuple(pair)
            if args.save:
                test_spec.write(pair[0].split('.')[0] + '\n')
            count += 1

        ttl_frames += int(pair[1])

    assert count == len(data)
