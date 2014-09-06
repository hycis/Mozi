

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='''get the specfiles for training validating and testing''')
parser.add_argument('--specname_path', metavar='PATH', required=True, help='the path to the specname npy file')
parser.add_argument('--train_valid_test_ratio', metavar='NUMBER STRING', required=True, help='''format '1 2 3' ''')

args = parser.parse_args()

name_path = args.specname_path
tvt_ratio = args.train_valid_test_ratio.strip().split(' ')
tvt_ratio = [int(s) for s in tvt_ratio]

assert len(tvt_ratio) == 3

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
        count += 1
    
    elif ttl_frames < valid_test_break:
        print 'valid: %s : %s'%tuple(pair)
        count += 1
    else:
        print 'test: %s : %s'%tuple(pair)
        count += 1
    
    ttl_frames += int(pair[1])

assert count == len(data)

