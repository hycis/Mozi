import numpy as np
import Image
import cPickle
from pynet.datasets.preprocessor import GCN
dir = '/Volumes/Storage/generated_specs/mgc'

base = '1113_1'

def cal_min():
    s = open(dir + '/%s.spec.f8'%base)
    s = np.fromfile(s, dtype='<f8')
    s = s.reshape((s.shape[0]/120, 120))

    m = open(dir + '/%s.mgc'%base)
    m = np.fromfile(m, dtype='<f4')
    m = m.reshape((m.shape[0]/120, 120))

    diff = m.shape[0] - s.shape[0]
    print 'diff', diff
    print 'mean s', (s**2).mean()
    print 'mean m', (s**2).mean()
    error = []
    min = 1000000
    min_i = 0

    if diff > 0:
        for i in range(diff):
            e =  ((m[i:i-diff] - s)**2).mean()
            if e < min:
                min = e
                min_i = i
            print i, e

        sp = Image.fromarray(s * 1000)
        sp.save(dir + '/%s.sp_orig.gif'%base)
        mgc = Image.fromarray(m[min_i:min_i-diff] * 1000)
        mgc.save(dir + '/%s.mgc_slice.gif'%base)

    elif diff < 0:
        diff = abs(diff)
        for i in range(diff):
            e =  ((s[i:i-diff] - m)**2).mean()
            if e < min:
                min = e
                min_i = i
            print i, e
        sp = Image.fromarray(s[min_i:min_i-diff] * 1000)
        sp.save(dir + '/%s.sp_slice.gif'%base)
        mgc = Image.fromarray(m * 1000)
        mgc.save(dir + '/%s.mgc_orig.gif'%base)

    print 'min', min_i, min



names = ['1112_1']
def draw_gif():
    for name in names:

        sp_fin = open(dir + '/%s.spec.f8'%name)
        sp = np.fromfile(sp_fin, dtype='<f8')
        sp = sp.reshape((sp.shape[0]/120, 120))
        sp = Image.fromarray(sp * 1000)
        sp.save(dir + '/%s.sp.gif'%name)

        mgc_fin = open(dir + '/%s.mgc'%name)
        mgc = np.fromfile(mgc_fin, dtype='<f4')
        mgc = mgc.reshape((mgc.shape[0]/120, 120))
        mgc = Image.fromarray(mgc * 1000)
        mgc.save(dir + '/%s.mgc.gif'%name)


def generate_spec(model, mlp, mgc, out_spec, fea_size):
    fin = open(model)
    print 'load ae..'
    ae = cPickle.load(fin)
    fin.close()

    fin = open(mlp)
    print 'load mlp..'
    mlp = cPickle.load(fin)
    fin.close()

    mgc_fin = open(mgc)
    print 'open ', mgc
    mgc = np.fromfile(mgc_fin, dtype='<f4')
    assert mgc.shape[0] % fea_size == 0
    mgc = mgc.reshape((mgc.shape[0]/fea_size, fea_size))
    print 'mgc shape ', mgc.shape
    decoded = ae.decode(mgc)

    normalizer = mlp.fprop(decoded)
    print 'normalizer shape ', normalizer.shape
    # import pdb
    # pdb.set_trace()
    warp_spec = decoded * normalizer
    fout = open(out_spec, 'wb')
    print 'saving.. '
    warp_spec.tofile(fout, format='<f8')

    mgc_fin.close()
    fout.close()
    print 'done'

def reconstruct_spec(mlp, spec, outfile):

    fin = open(mlp)
    print 'load mlp..'
    mlp = cPickle.load(fin)
    fin.close()

    warp_spec = open(spec)
    warp_spec = np.fromfile(warp_spec, dtype='<f4')
    assert warp_spec.shape[0] % 2049 == 0
    warp_spec = warp_spec.reshape((warp_spec.shape[0]/2049, 2049))

    gcn = GCN()
    out = gcn.apply(warp_spec)

    out = out * mlp.fprop(out)

    output_file = open(outfile, 'wb')
    out.tofile(output_file, format='<f4')

def mask_spec(spec, dir):
    warp_fin = open(spec)
    warp_spec = np.fromfile(warp_fin, dtype='<f4')
    warp_fin.close()
    rd = np.random.binomial(1, 0.5, warp_spec.shape)
    mask_spec = warp_spec * rd
    fout = open(dir + '/2696_1_mask.spec.f4', 'wb')
    mask_spec = mask_spec.astype('f4')
    mask_spec.tofile(fout, format='<f4')
    fout.close()


    # assert warp_spec.shape[0] % 2049 == 0
    # warp_spec = warp_spec.reshape((warp_spec.shape[0]/2049, 2049))
    #
    # warp_img = Image.fromarray(warp_spec * 1)
    # warp_img.save(dir + '/orig.gif')
    #
    # print warp_spec.shape
    # rd = np.random.binomial(1, 0.5, warp_spec.shape)
    # mask_spec = rd * warp_spec
    # mask_img = Image.fromarray(mask_spec * 1)
    # mask_img.save(dir + '/mask.gif')

if __name__ == '__main__':
    # cal_min()
    # draw_gif()
    # bases = ['1115_1', '1116_1', '1117_1', '1118_1', '1119_1']
    # for base in bases:
    #     model='/home/smg/zhenzhou/Pynet/save/log/AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy_20140919_2220_41651965/cpu_model.pkl'
    #     mlp='/home/smg/zhenzhou/Pynet/save/log/AE0928_Warp_Laura_Blocks_GCN_Mapping_20140929_2352_20882069/cpu_model.pkl'
    #     mgc='/home/smg/takaki/DNN/Zhenzhou/20140925/120_warp_n/%s.mgc'%base
    #     fea_size=120
    #     output_spec='/home/smg/zhenzhou/special/%s.spec.f8'%base
    #     generate_spec(model, mlp, mgc, output_spec, fea_size)

    # mlp = '/Volumes/Storage/models/mapper/AE0928_Warp_Laura_Blocks_GCN_Mapping_20140929_2352_20882069/cpu_model.pkl'
    spec = '/Volumes/Storage/generated_specs/Laura/orig/2696_1.spec'
    # outfile = '/Volumes/Storage/models/mapper/1119_1.spec.map2.f4'
    # reconstruct_spec(mlp, spec, outfile)
    dir = '/Volumes/Storage/Dropbox/project_nii'
    mask_spec(spec, dir)
