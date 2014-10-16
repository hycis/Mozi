import argparse
import os
import socket

parser = argparse.ArgumentParser(description='''Sync a model between different servers''')

parser.add_argument('--from_to', nargs='+', help='''sync from [gill | biaree | nii | udem | helios]
                                                to [gill | biaree | udem | helios]''')
parser.add_argument('--model', metavar='Name', help='''model to be sync''')

args = parser.parse_args()
assert len(args.from_to) == 2, 'more than two arguments for --from_to'

gill='hycis@guillimin.clumeq.ca:/sb/project/jvb-000-aa/zhenzhou/Pynet/save/log'
biaree='hycis@briaree.calculquebec.ca:/RQexec/hycis/Pynet/save/log'
udem='wuzhen@frontal07.iro.umontreal.ca:~/Pynet/save/log'
nii='zhenzhou@136.187.97.216:~/Pynet/save/log'
helios='hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/Pynet/save/log'

hostname = socket.gethostname()

tbl = {'nii': 'cn01051002.ecloud.nii.ac.jp', 'helios' : 'helios-login1',
        'biaree': 'briaree1', 'gill' : 'lg-1r17-n04'}

if tbl[args.from_to[0]] == hostname:
    from_server = '%s/%s'%(locals()[args.from_to[0]].split(':')[-1], args.model)
    to_server = locals()[args.from_to[1]]

elif tbl[args.from_to[1]] == hostname:
    from_server = '%s/%s'%(locals()[args.from_to[0]], args.model)
    to_server = locals()[args.from_to[1]].split(':')[-1]
else:
    raise Exception('hostname not in gill biaree nii udem or helios')

print 'rsync -rvu %s %s'%(from_server, to_server)

os.system('rsync -rvu %s %s'%(from_server, to_server))
