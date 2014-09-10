from hps.models import *
import os


def setEnv():

    NNdir = os.path.dirname(os.path.realpath(__file__))
    NNdir = os.path.dirname(NNdir)
    NNdir = os.path.dirname(NNdir)

    if not os.getenv('PYNET_DATA_PATH'):
        os.environ['PYNET_DATA_PATH'] = NNdir + '/data'

    if not os.getenv('PYNET_DATABASE_PATH'):
        os.environ['PYNET_DATABASE_PATH'] = NNdir + '/database'
        if not os.path.exists(os.environ['PYNET_DATABASE_PATH']):
            os.mkdir(os.environ['PYNET_DATABASE_PATH'])

    if not os.getenv('PYNET_SAVE_PATH'):
        os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'
        if not os.path.exists(os.environ['PYNET_SAVE_PATH']):
            os.mkdir(os.environ['PYNET_SAVE_PATH'])


    print('PYNET_DATA_PATH = ' + os.environ['PYNET_DATA_PATH'])
    print('PYNET_SAVE_PATH = ' + os.environ['PYNET_SAVE_PATH'])
    print('PYNET_DATABASE_PATH = ' + os.environ['PYNET_DATABASE_PATH'])


def AE_exp(state, channel):

    setEnv()
    hps = AE(state)
    hps.run()

    return channel.COMPLETE

def AE_Testing_exp(state, channel):
    setEnv()
    hps = AE_Testing(state)
    hps.run()

    return channel.COMPLETE

def Laura_exp(state, channel):
    setEnv()
    hps = Laura(state)
    hps.run()

    return channel.COMPLETE

def Laura_Two_Layers_exp(state, channel):
    setEnv()
    hps = Laura_Two_Layers(state)
    hps.run()

    return channel.COMPLETE

def Laura_Three_Layers_exp(state, channel):
    setEnv()
    hps = Laura_Three_Layers(state)
    hps.run()

    return channel.COMPLETE

def Laura_Two_Layers_No_Transpose_exp(state, channel):
    setEnv()
    hps = Laura_Two_Layers_No_Transpose(state)
    hps.run()

    return channel.COMPLETE