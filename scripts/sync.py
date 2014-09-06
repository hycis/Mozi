
import argparse
import os

parser = argparse.ArgumentParser(description='''Sync smartNN with biaree, udem or nii server''')

parser.add_argument('--udem', action='store_true', help='''sync with udem''')
parser.add_argument('--nii', action='store_true', help='''sync with nii''')
parser.add_argument('--biaree', action='store_true', help='''sync with biaree''')
parser.add_argument('--image', action='store_true', help='''sync with the image folder''')
parser.add_argument('--scratch', action='store_true', help='''sync with the scratch folder''')
parser.add_argument('--nik', action='store_true', help='''sync with the nikopia''')
parser.add_argument('--lua', action='store_true', help='''sync with the lua folder''')
parser.add_argument('--rd', action='store_true', help='''sync with the research demo folder''')
parser.add_argument('--pynet', action='store_true', help='''sync with the pynet folder''')
parser.add_argument('--gill', action='store_true', help='''sync with gillimin''')
parser.add_argument('--smartnn', action='store_true', help='''sync with smartnn''')
parser.add_argument('--helios', action='store_true', help='''sync with helios''')


args = parser.parse_args()

source = '/Volumes/Storage/Dropbox/CodingProjects'
exclude = '--exclude-from=%s/smartNN/exclude.txt'%source
lua_package = '/Volumes/Storage/lua_packages'
lua_exclude = '--exclude-from=%s/exclude.txt'%lua_package


if args.gill:
	if args.pynet:
	    os.system("rsync -rvu %s %s/Pynet hycis@guillimin.clumeq.ca:/sb/project/jvb-000-aa/zhenzhou"%(exclude, source))
	
	elif args.smartnn:
		os.system("rsync -rvu %s %s/smartNN hycis@guillimin.clumeq.ca:/sb/project/jvb-000-aa/zhenzhou"%(exclude, source))  

elif args.nii:
	if args.image:
		os.system("rsync -rvu zhenzhou@136.187.97.216:~/smartNN/save/images \
					%s/smartNN/save/images/nii"%source)
					
	elif args.pynet:
	    os.system("rsync -rvu %s %s/Pynet zhenzhou@136.187.97.216:~/"%(exclude, source))
	
	elif args.smartnn:
		os.system("rsync -rvu %s %s/smartNN zhenzhou@136.187.97.216:~/"%(exclude, source))

elif args.biaree:
	if args.image:
		os.system("rsync -rvu hycis@briaree.calculquebec.ca:~/smartNN/save/images \
					%s/smartNN/save/images/biaree"%source)

	elif args.pynet:
		os.system("rsync -rvu %s %s/Pynet hycis@briaree.calculquebec.ca:/RQexec/hycis"%(exclude, source))
    
	elif args.smartnn:
		os.system("rsync -rvu %s %s/smartNN hycis@briaree.calculquebec.ca:/RQexec/hycis"%(exclude, source))

elif args.udem:
    if args.rd:
	    os.system("rsync -rvu %s /Volumes/Storage/VCTK/Research-Demo \
                wuzhen@elisa2.iro.umontreal.ca:/data/lisa/exp/wuzhen/nii/VoiceCloneCommercial2/"%exclude)

    elif args.lua:
	    os.system("rsync -rvu %s /Volumes/Storage/lua_packages \
                wuzhen@frontal07.iro.umontreal.ca:/data/lisa/exp/wuzhen/"%lua_exclude)
    
    elif args.pynet:
        os.system("rsync -rvu %s %s/Pynet \
                wuzhen@frontal07.iro.umontreal.ca:~/"%(exclude, source))
    elif args.smartnn:
        os.system("rsync -rvu %s %s/smartNN \
                wuzhen@frontal07.iro.umontreal.ca:~/"%(exclude, source))

elif args.helios:
    if args.pynet:
        os.system("rsync -rvu %s %s/Pynet \
                hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis"%(exclude, source))  

elif args.nik:
    if args.lua:
	    os.system("rsync -rvu %s /Volumes/Storage/lua_packages \
                zhenzhou@nikopia.net:/home/zhenzhou/"%lua_exclude)

else:
	raise ValueError('options is neither --udem | --nii | --biaree | --udemrd | --udemlua | --image | --niklua | --pynet')
