import matplotlib
matplotlib.use('Agg')
from pynet.utils.utils import make_one_hot
from pynet.datasets.dataset_noise import Gaussian
from visualize_data import *


def sample_shuffle_data(example_paths, target_paths, accepted_rate=0.1):
    '''
    randomly take certain fraction of the samples from each file and mix them together.
    '''
    examples = glob.glob(example_paths)
    examples.sort()
    targets = glob.glob(target_paths)
    targets.sort()

    sample_X = []
    sample_y = []

    for exp, tar in zip(examples, targets):
        with open(exp) as exp_fin, open(tar) as tar_fin:
            print 'open', exp, tar
            X = np.load(exp_fin)
            y = np.load(tar_fin)

            for a, b in zip(X, y):
                x = random.uniform(0, 1)
                if x < accepted_rate:
                    sample_X.append(a)
                    sample_y.append(b)
            print 'sample_X, sample_y:', len(sample_X), len(sample_y)


    idxs = np.arange(len(sample_X))
    np.random.shuffle(idxs)
    np.random.shuffle(idxs)
    sample_X = np.asarray(sample_X)
    sample_y = np.asarray(sample_y)
    return sample_X[idxs], sample_y[idxs]

def random_merge_split(exp_path1, exp_path2, target_path1, target_path2,
                       exp_save1, exp_save2, target_save1, target_save2):
    '''
    randomly take two files, merge, shuffle the merge idxs, then split
    '''
    try:
        with open(exp_path1) as fe1, open(exp_path2) as fe2, \
            open(target_path1) as ft1, open(target_path2) as ft2:


            x1 = np.load(fe1)
            x2 = np.load(fe2)

            y1 = np.load(ft1)
            y2 = np.load(ft2)

            merge_X = np.concatenate((x1, x2), axis=0)
            merge_y = np.concatenate((y1, y2), axis=0)

            assert len(merge_X) == len(merge_y)

            idxs = np.arange(len(merge_X))
            print '..shuffling idxs'
            np.random.shuffle(idxs)
            np.random.shuffle(idxs)
            merge_X = merge_X[idxs]
            merge_y = merge_y[idxs]

        with open(exp_save1, 'wb') as oe1, open(exp_save2, 'wb') as oe2, \
            open(target_save1, 'wb') as ot1, open(target_save2, 'wb') as ot2:
            print '..spliting and saving'
            split_X_ls = np.array_split(merge_X, 2)
            split_y_ls = np.array_split(merge_y, 2)
            np.save(oe1, split_X_ls[0])
            np.save(oe2, split_X_ls[1])
            np.save(ot1, split_y_ls[0])
            np.save(ot2, split_y_ls[1])
            print '..done!'
    except:
        print '..error opening'


def random_shuffles(exp_paths, target_paths, num_of_mix_rounds=100):
    exp_paths = sorted(glob.glob(exp_paths))
    target_paths = sorted(glob.glob(target_paths))

    assert len(exp_paths) == len(target_paths)

    population = xrange(len(exp_paths))
    for i in xrange(num_of_mix_rounds):
        print '..mixing round {}/{}'.format(i+1, num_of_mix_rounds)
        [id1, id2] = random.sample(population, 2)
        print 'exp1', exp_paths[id1]
        print 'target1', target_paths[id1]
        print 'exp2', exp_paths[id2]
        print 'target2', target_paths[id2]
        random_merge_split(exp_paths[id1], exp_paths[id2], target_paths[id1], target_paths[id2],
                           exp_paths[id1], exp_paths[id2], target_paths[id1], target_paths[id2])

    print '..all done!'



def save_data(X, y, save_dir):
    print 'saving'
    with open("%s/sample_X.npy"%save_dir, 'wb') as Xout, \
        open("%s/sample_y.npy"%save_dir, 'wb') as yout:
        np.save(Xout, X)
        np.save(yout, y)
        print 'saving done!'


def save_gaussian_one_hot(y, save_dir, std):
    print 'saving'
    one_hot_y = make_one_hot(y, 1998)
    gaussian = Gaussian(std=std)
    one_hot_y = gaussian.apply(one_hot_y)
    with open("%s/sample_y_onehot_gaussian_noise_std%s.npy"%(save_dir,str(std)), 'wb') as yout:
        np.save(yout, one_hot_y)
        print 'saving done!'


def make_gaussian_one_hot(std):
    example_paths = '/home/stuwzhz/datasets/spectral-features/npy2/ClnDNN_NoisyFeat.post_00*.npy'
    target_paths =  '/home/stuwzhz/datasets/spectral-features/npy2/clean.pdf_00*.npy'
    save_dir = os.path.dirname(example_paths)
    sample_X, sample_y = sample_shuffle_data(example_paths, target_paths)
    save_data(sample_X, sample_y, save_dir)
    save_gaussian_one_hot(sample_y, save_dir, std=std)

def make_fig(std):
    class_label = 49
    example_paths = '/home/stuwzhz/datasets/spectral-features/npy2/sample_y_onehot_gaussian_noise.npy'
    target_paths =  '/home/stuwzhz/datasets/spectral-features/npy2/sample_y.npy'
    fig_dir = os.path.dirname(example_paths)
    X, colors = sample_data(example_paths, target_paths, class_label, accepted_rate=1)
    angles, lengths = cal_angles(X, class_label)
    plt = plot_angles(angles, lengths, colors, class_label)
    plt.savefig('%s/gaussian_noise_std%s_class_%s.png'%(fig_dir, str(std), class_label))

def main1():
    example_paths = '/scratch/stuwzhz/dataset/npy/ClnDNN_NoisyFeat.post_*.npy'
    target_paths =  '/scratch/stuwzhz/dataset/npy/clean.pdf_*.npy'
    num_of_rounds = 300
    random_shuffles(example_paths, target_paths, num_of_rounds)

def main2():
    example_paths = '/scratch/stuwzhz/dataset/npy/ClnDNN_NoisyFeat.post_*.npy'
    target_paths =  '/scratch/stuwzhz/dataset/npy/clean.pdf_*.npy'

    examples = sorted(glob.glob(example_paths))
    targets = sorted(glob.glob(target_paths))

    for exp, tar in zip(examples, targets):
        try:
            with open(exp) as fe, open(tar) as ft:
                print exp
                print tar
                x = np.load(fe)
                y = np.load(ft)
        except:
            print 'error loading'
            print 'replacing', exp, tar
            exp_b = os.path.basename(exp)
            tar_b = os.path.basename(tar)

            data_dir = '/home/stuwzhz/datasets/spectral-features/npy2'
            scrat_dir = '/scratch/stuwzhz/dataset/npy'

            os.system("cp {}/{} {}".format(data_dir, exp_b, scrat_dir))
            os.system("cp {}/{} {}".format(data_dir, tar_b, scrat_dir))


if __name__ == '__main__':
    # make_gaussian_one_hot(std=0.3)
    # make_fig(std=0.5)
    main1()
