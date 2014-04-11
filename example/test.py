
import numpy as np
import glob
import itertools

def savenpy(folder_path):

    
    files = glob.glob(folder_path + '/*.spec')
    size = len(files)
    data = []
    count = 0
    for f in files:
        with open(f) as fb:
            clip = np.fromfile(fb, dtype='<f4', count=-1)
            data.extend(clip)

        print(str(count) + '/' + str(size) + '..done '  + f)
        
        count += 1

    with open(folder_path + '/Laura.npy', 'wb') as f:
        np.save(f, data)

    print('all finished successfully')

def processnpy(filepath):
    
    with np.load(filepath, 'r') as obj:

        import pdb
        pdb.set_trace()
        
def testpreproc(path):
    from smartNN.datasets.preprocessor import Scale
    import numpy as np
    pre = Scale()
    with open(path) as f:
        X = np.load(f)
    X = pre.apply(X)
    import pdb
    pdb.set_trace()
    

if __name__ == '__main__':

#     savenpy('/RQusagers/hycis/smartNN/data/Laura')
#     testpreproc('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy')
    testpreproc('/data/lisa/exp/wuzhen/smartNN/data/p276/p276.npy')
#     processnpy('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy')  

