

import numpy as np
import cPickle
import os

def make_posterior_npy(filename, save_dir, splits):

    basename = os.path.basename(filename)
    with open("%s/clean.pdf_fname_count.pkl"%save_dir) as fname:
        fnames_ls = cPickle.load(fname)

    num_file = len(fnames_ls)

    file_count = 0
    part = 0
    with open(filename) as fin:

        print 'num of files', num_file
        files_per_split = num_file / splits
        print 'file per split', files_per_split

        npy = []
        in_file = False
        frames_count = 0
        for line in fin:
            toks = line.strip().split()
            if toks[0][:-2] == fnames_ls[file_count][0][:-2]:
                print 'processing file:', file_count, toks[0]
                in_file = True

            elif toks[-1] == ']':
                toks.pop(-1)
                npy.append(toks)
                frames_count += fnames_ls[file_count][1]
                assert len(npy) == frames_count, \
                    'frame_count %s not equal to fnames_ls count %s'%(len(npy), frames_count)

                if (file_count+1) % files_per_split == 0:
                    print 'saving %s/%s_%.3d.npy, num of frames %d'%(save_dir, basename, part, len(npy))
                    with open('%s/%s_%.3d.npy'%(save_dir, basename, part), 'wb') as Xout:
                        np.save(Xout, np.asarray(npy).astype('f4'))
                    part += 1
                    npy = []
                    frames_count = 0

                file_count += 1
                in_file = False

            elif in_file:
                npy.append(toks)

            else:
                raise Exception('error: frame not in file!')

        if len(npy) > 0:
            print 'saving %s/%s_%.3d.npy, num of frames %d'%(save_dir, basename, part, len(npy))
            with open('%s/%s_%.3d.npy'%(save_dir, basename, part), 'wb') as Xout:
                np.save(Xout, np.asarray(npy).astype('f4'))



# def make_random_posterior_npy(filename, save_dir, splits):
#     basename = os.path.basename(filename)
#
#     with open("%s/clean.pdf_filename_framecount.pkl"%save_dir) as fname:
#         filename_framecount_ls = cPickle.load(fname)
#
#     num_file = len(filename_framecount_ls)
#
#     file_count_ls = []
#     file_count = 0
#     frame_count = 0
#     with open(filename) as fin:
#         print 'num of files', num_file
#         files_per_split = num_file / splits
#         print 'file per split', files_per_split
#         in_file = False
#         fname = None
#         npy = []
#         for line in fin:
#             toks = line.strip().split()
#             if toks[0][:-2] == filename_framecount_ls[file_count][0][:-2]:
#                 print 'processing file:', file_count, toks[0]
#                 in_file = True
#                 fname = toks[0]
#
#             elif toks[-1] == ']':
#                 toks.pop(-1)
#                 # npy.append(toks)
#
#                 if (file_count+1) % files_per_split == 0:
#                     print 'saving %s/%s_%.3d.npy, num of frames %d'%(save_dir, basename, part, len(npy))
#                     with open('%s/%s_%.3d.npy'%(save_dir, basename, part), 'wb') as Xout:
#                         np.save(Xout, np.asarray(npy).astype('f4'))
#
#                 frame_count += 1
#                 assert frame_count == filename_framecount_ls[file_count][1]
#                 in_file = False
#                 file_count_ls.append((fname, frame_count))
#                 file_count += 1
#                 frame_count = 0
#                 fname = None
#                 # if file_count == 100:
#                 #     import pdb
#                 #     pdb.set_trace()
#
#             elif in_file:
#                 # npy.append(toks)
#                 frame_count += 1
#
#             else:
#                 raise Exception('error: frame not in file!')
#     import pdb
#     pdb.set_trace()






def make_posterior_filename_framecount(filename, save_dir, splits):

    basename = os.path.basename(filename)

    with open("%s/clean.pdf_fname_count.pkl"%save_dir) as fname:
        fnames_ls = cPickle.load(fname)

    num_file = len(fnames_ls)
    file_count_ls = []
    file_count = 0
    frame_count = 0
    with open(filename) as fin:

        print 'num of files', num_file
        files_per_split = num_file / splits
        print 'file per split', files_per_split

        in_file = False
        fname = None
        for line in fin:
            toks = line.strip().split()
            if toks[0][:-2] == fnames_ls[file_count][0][:-2]:
                print 'processing file:', file_count, toks[0]
                in_file = True
                fname = toks[0]

            elif toks[-1] == ']':
                frame_count += 1
                assert frame_count == fnames_ls[file_count][1]
                in_file = False
                file_count_ls.append((fname, frame_count))
                file_count += 1
                frame_count = 0
                fname = None

            elif in_file:
                frame_count += 1

            else:
                raise Exception('error: frame not in file!')


        print 'saving %s/%s_fname_count.pkl'%(save_dir, basename)
        with open('%s/%s_fname_count.pkl'%(save_dir, basename), 'wb') as Xout:
            cPickle.dump(file_count_ls, Xout)

        print str(fnames_ls)
        print str(file_count_ls)


def make_labels_npy(filename, save_dir, splits):

    file_count_ls = []

    num_file = 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    basename = os.path.basename(filename)

    with open(filename) as fin:

        for line in fin:
            num_file += 1
        fin.seek(0)

        print 'num of files', num_file
        files_per_split = num_file / splits
        print 'file per split', files_per_split

        count = 1
        file_count = 0
        npy = []
        for line in fin:

            toks = line.strip().split()
            if count < files_per_split:
                file_count_ls.append((toks[0], len(toks)-1))
                npy += toks[1:]
                count += 1

            else:
                print 'saving split, num_frames', file_count, len(npy)
                with open("%s/%s_%.3d.npy"%(save_dir, basename, file_count), 'wb') as fout:
                    file_count_ls.append((toks[0], len(toks)-1))
                    npy += toks[1:]
                    np.save(fout, np.asarray(npy).astype('f4'))
                    count = 1
                    file_count += 1
                    npy = []

        if len(npy) > 0:
            print 'saving split, num_frames', file_count, len(npy)
            with open("%s/%s_%.3d.npy"%(save_dir, basename, file_count), 'wb') as fout:
                np.save(fout, np.asarray(npy).astype('f4'))

        with open("%s/%s_fname_count.pkl"%(save_dir, basename), 'wb') as fname:
            cPickle.dump(file_count_ls, fname)


if __name__ == '__main__':
    # make_labels_npy(filename='/home/stuwzhz/datasets/spectral-features/clean.pdf',
    #             save_dir='/home/stuwzhz/datasets/spectral-features/data_npy', splits=1)

    # make_posterior_filename_framecount(filename='/home/stuwzhz/datasets/spectral-features/train_si84_clean_fbank_cmn_d_dd',
    #                             save_dir='/home/stuwzhz/datasets/spectral-features/data_npy', splits=1)

    make_posterior_npy(filename='/home/stuwzhz/datasets/spectral-features/train_si84_clean_fbank_cmn_d_dd',
                        save_dir='/home/stuwzhz/datasets/spectral-features/data_npy', splits=1)
