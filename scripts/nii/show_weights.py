import matplotlib
matplotlib.use('Agg')
import cPickle
from matplotlib import pyplot as plt

import os



data_dir = '/Volumes/Storage/models'
img_save_dir = '/Volumes/Storage/weights'

models = [('clean', 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_clean_continue_20141110_1235_21624029'),
          ('gaussian', 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_gaussian_continue_20141110_1250_49502872'),
          ('maskout', 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_continue_20141110_1251_56190462'),
          ('blackout', 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_blackout_continue_20141110_1249_12963320'),
          ('batchout', 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_batchout_continue_20141111_0957_22484008')]


for name, md in models:
    with open("{}/{}/model.pkl".format(data_dir, md)) as fin:
        print name
        model = cPickle.load(fin)
        W = model.layers[0].W.get_value()
        plt.hist(W.flatten(), 100, range=(W.min(), W.max()), fc='k', ec='k')
        plt.xlabel('weights')
        plt.ylabel('frequency of occurrence')
        plt.title('Histogram of Weights')
        plt.savefig('{}/{}_hist.pdf'.format(img_save_dir, name))
        plt.close()
        plt.imshow(W, vmin=-0.05, vmax=0.05)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel('bottleneck dim', fontsize=10)
        plt.ylabel('input dim', fontsize=12)
        plt.colorbar()
        plt.savefig('{}/{}.pdf'.format(img_save_dir, name), bbox_inches='tight')
        plt.close()

# for name, md in models:
#     os.system('python sync_model.py --from_to helios home --model {}'.format(md))
