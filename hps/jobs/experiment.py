import os
import importlib

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
        os.environ['PYNET_SAVE_PATH'] = NNdir + '/save/log'
        if not os.path.exists(os.environ['PYNET_SAVE_PATH']):
            os.mkdir(os.environ['PYNET_SAVE_PATH'])


    print('PYNET_DATA_PATH = ' + os.environ['PYNET_DATA_PATH'])
    print('PYNET_SAVE_PATH = ' + os.environ['PYNET_SAVE_PATH'])
    print('PYNET_DATABASE_PATH = ' + os.environ['PYNET_DATABASE_PATH'])


def job(state, channel):
    setEnv()
    module = importlib.import_module("hps.models.%s"%state.module_name)
    obj = getattr(module, state.module_name)(state)
    obj.run()
    return channel.COMPLETE
