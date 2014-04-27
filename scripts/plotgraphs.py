import numpy as np
from smartNN.utils.utils import tile_raster_graphs

dir = '/Users/Hycis/Desktop/compare/'

dct_reconstruct = np.fromfile(dir + 'p276_002.spec.double', dtype='f8')
orig = np.fromfile(dir + 'p276_002Orig.spec', dtype='f4')
ae_reconstruct = np.fromfile(dir + 'p276_002.spec.f8', dtype='f8')

dct_reconstruct = dct_reconstruct.reshape(dct_reconstruct.shape[0]/2049, 2049)
ae_reconstruct = ae_reconstruct.reshape(ae_reconstruct.shape[0]/2049, 2049)
orig = orig.reshape(orig.shape[0]/2049, 2049)



plt = tile_raster_graphs(dct_reconstruct[100:110], orig[100:110], ae_reconstruct[100:110], 
                        tile_shape=(10,1), tile_spacing=(0.1,0.1), slice=(20,80), axis=None, legend=True)
plt.show()