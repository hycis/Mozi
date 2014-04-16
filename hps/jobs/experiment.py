from hps.models import AE, AE_Two_Layers
import os

def AE_exp(state, channel):
    
    NNdir = os.path.dirname(os.path.realpath(__file__))
    NNdir = os.path.dirname(NNdir)
    NNdir = os.path.dirname(NNdir)
    
    if not os.getenv('smartNN_DATA_PATH'):
        os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

    if not os.getenv('smartNN_DATABASE_PATH'):
        os.environ['smartNN_DATABASE_PATH'] = NNdir + '/database'

    if not os.getenv('smartNN_SAVE_PATH'):
        os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'
    
    print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
    print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])
    print('smartNN_DATABASE_PATH = ' + os.environ['smartNN_DATABASE_PATH'])
    hps = AE(state)
    hps.run()
    
    return channel.COMPLETE

def AE_Two_Layers_exp(state, channel):
    
    NNdir = os.path.dirname(os.path.realpath(__file__))
    NNdir = os.path.dirname(NNdir)
    NNdir = os.path.dirname(NNdir)
    
    if not os.getenv('smartNN_DATA_PATH'):
        os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

    if not os.getenv('smartNN_DATABASE_PATH'):
        os.environ['smartNN_DATABASE_PATH'] = NNdir + '/database'

    if not os.getenv('smartNN_SAVE_PATH'):
        os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'
    
    print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
    print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])
    print('smartNN_DATABASE_PATH = ' + os.environ['smartNN_DATABASE_PATH'])

    hps = AE_Two_Layers(state)
    hps.run()
    
    return channel.COMPLETE