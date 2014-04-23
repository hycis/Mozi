import cPickle
import os
import glob
import argparse
import numpy as np
from smartNN.datasets.preprocessor import GCN
import argparse


parser = argparse.ArgumentParser(description='''Reconstruction of data from model''')
parser.add_argument('--model', help='''the model use for reconstruction''')
parser.add_argument('--preprocessor', default='spec', help='''extension of the spec files''')
parser.add_argument('--splits', default=1, type=int, help='''number of splits for the merged spec files''')
parser.add_argument('--input_spec_dtype',  default='f4', help='''dtype of the input spec files f4|f8''')

args = parser.parse_args()


thisfile_path = os.path.realpath(__file__)
thisfile_dir = os.path.dirname(thisfile_path)
NNdir = os.path.dirname(thisfile_dir)
best_model = 'AE15Double_GCN_20140415_1404_50336696'


print 'loading model ' + best_model
with open(NNdir + '/save/log/' + best_model + '/model.pkl', 'rb') as pkl:
    model = cPickle.load(pkl)

print 'opening ' + NNdir + '/data/p276/p276_data_000.npy'

data_file = NNdir + '/data/p276/p276_data_000.npy'
obj = open(data_file, 'rb')
data = np.load(obj)

print 'data.shape: ' + data.shape

print '..applying preprocessing'
proc = GCN()
proc_data = proc.apply(data)
print '..fprop'
output = model.fprop(proc_data)
print '..invert'
inverted_data = proc.invert(output)

print '..save inverted'
outfile = open(data_file + '.out', 'wb')
np.save(outfile, inverted_data)
obj.close()
outfile.close()
print '..completed successfully'

