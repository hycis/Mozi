import argparse
import os

parser = argparse.ArgumentParser(description='''Sync a model between different servers''')

parser.add_argument('--from_to', nargs='+', help='''sync from [gill | biaree | nii | udem] to [gill | biaree | udem]''')
parser.add_argument('--model', metavar='Name', help='''model to be sync''')

args = parser.parse_args()
assert len(args.from_to) == 2, 'more than two arguments for --from_to'

gill='hycis@guillimin.clumeq.ca:/sb/project/jvb-000-aa/zhenzhou/Pynet/save/log'
biaree='hycis@briaree.calculquebec.ca:/RQexec/hycis/Pynet/save/log'
udem='wuzhen@frontal07.iro.umontreal.ca:~/Pynet/save/log'
nii='zhenzhou@136.187.97.216:~/Pynet/save/log'

from_server = '%s/%s'%(locals()[args.from_to[0]].split(':')[-1], args.model) 
to_server = locals()[args.from_to[1]]

os.system('rsync -rvu %s %s'%(from_server, to_server))