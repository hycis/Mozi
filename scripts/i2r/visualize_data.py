import matplotlib
matplotlib.use('Agg')

import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import subplot, scatter



def sample_data(example_paths, target_paths, class_label, accepted_rate=0.1):
    examples = glob.glob(example_paths)
    examples.sort()
    targets = glob.glob(target_paths)
    targets.sort()
    block = []
    colors = []
    count = 0
    total_count = 0

    for exp, tar in zip(examples, targets):
        with open(exp) as exp_fin, open(tar) as tar_fin:
            print 'open', exp, tar
            X = np.load(exp_fin)
            y = np.load(tar_fin)

            for i, val in enumerate(y):
                if val == class_label:
                    total_count += 1
                    x = random.uniform(0, 1)
                    if x < accepted_rate:
                        block.append(X[i])
                        if np.argmax(X[i]) == class_label:
                            colors.append(0)
                        else:
                            colors.append(1)
                        count += 1
            print 'count/total_count', count, total_count

    return np.asarray(block), np.asarray(colors)


def cal_angles(X, class_label):
    length = np.sqrt(np.sum(X**2, axis=1))
    cos_theta = X[:, class_label] / length
    theta = np.arccos(cos_theta)
    return theta, length


def plot_angles(theta, length, colors, class_label):
    ax = subplot(111, polar=True)
    c = scatter(theta, length, c=colors, label='class %s'%str(class_label))
    ax.grid(True)
    c.set_alpha(0.75)
    ax.set_title("Plot of noisy posterior for class %s"%class_label, va='bottom')
    return plt

def visualize_posterior():
    from pynet.datasets.i2r import I2R_Posterior_Gaussian_Noisy_Sample
    data = I2R_Posterior_Gaussian_Noisy_Sample()
    y = data.get_train().y
    y1 = np.sum(y, axis=0)
    y2 = np.sort(y1)
    # import pdb
    # pdb.set_trace()
    plt.plot(y1)
    plt.savefig("/home/stuwzhz/Pynet/hps/unsorted_fig.png")


def main():
    class_label = 1000
    example_paths = '/home/stuwzhz/datasets/spectral-features/npy2/ClnDNN_CleanFeat.post_00*.npy'
    target_paths =  '/home/stuwzhz/datasets/spectral-features/npy2/clean.pdf_00*.npy'
    fig_dir = os.path.dirname(example_paths)
    X, colors = sample_data(example_paths, target_paths, class_label)
    angles, lengths = cal_angles(X, class_label)
    plt = plot_angles(angles, lengths, colors, class_label)
    plt.savefig('%s/clean_class_%s.png'%(fig_dir, class_label))


if __name__ == '__main__':
    # main()
    visualize_posterior()
