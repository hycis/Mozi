from hps.hps import AE_HPS
import os

def experiment(state, channel):
    
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

    hps = AE_HPS(state)
    hps.run()
    
    return channel.COMPLETE