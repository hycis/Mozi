
import argparse
import cPickle
import glob
import os
import numpy as np
import pynet.datasets.preprocessor as procs
import pynet.datasets.dataset_noise as noises

parser = argparse.ArgumentParser(description='preprocess a spec and save.')
parser.add_argument('--spec', metavar='PATH', help='spec to be preprocessed')
parser.add_argument('--dtype', metavar='TYPE', help='dtype of spec')
parser.add_argument('--preprocessor', metavar='NAME', help='name of the preprocessor')
parser.add_argument('--noise', metavar='NAME', help='type of noise')
parser.add_argument('--frame_size', metavar='NUMBER', default=2049, help='frame size of each spec')

args = parser.parse_args()

with open(args.spec) as fin:
    clip = np.fromfile(fin, dtype='<%s'%args.dtype, count=-1)
    clip_size = clip.shape[0]
    assert clip_size % args.frame_size == 0, 'frame size is not multiple of length of spec'
    clip = clip.reshape((clip_size/args.frame_size, args.frame_size))

    proc = getattr(procs, args.preprocessor)()
    print 'preprocessor:', proc.__class__.__name__
    clip = proc.apply(clip)

    noise = getattr(noises, args.noise)()
    print 'adding noise:', noise.__class__.__name__
    clip = noise.apply(clip)

    print 'inverting'
    clip = proc.invert(clip)

    bname = os.path.basename(args.spec)
    dname = os.path.dirname(args.spec)

    clip = clip.reshape(clip_size)
    clip = clip.astype(args.dtype)
    print 'saving'
    clip.tofile(dname + '/%s_%s'%(args.noise,bname), format=args.dtype)

print 'done!'
