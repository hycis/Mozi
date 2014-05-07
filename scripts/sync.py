
import argparse
import os

parser = argparse.ArgumentParser(description='''Sync smartNN with biaree, udem or nii server''')

parser.add_argument('--udem', action='store_true', help='''sync with udem''')

parser.add_argument('--nii', action='store_true', help='''sync with nii''')

parser.add_argument('--biaree', action='store_true', help='''sync with biaree''')

parser.add_argument('--image', action='store_true', help='''sync with the image folder''')

parser.add_argument('--udemrd', action='store_true', help='''sync with the udem research demo folder''')


args = parser.parse_args()

source = '/Volumes/Storage/Dropbox/CodingProjects/smartNN'
exclude = '--exclude-from=%s/exclude.txt'%source
if args.udem:
	if args.image:
		os.system("rsync -rvu wuzhen@elisa2.iro.umontreal.ca:/data/lisa/exp/wuzhen/smartNN/save/images \
					%s/save/images/udem"%source) 
	else:
		os.system("rsync -rvu %s %s wuzhen@elisa2.iro.umontreal.ca:/data/lisa/exp/wuzhen"%(exclude, source)) 

elif args.nii:
	if args.image:
		os.system("rsync -rvu zhenzhou@136.187.97.216:~/smartNN/save/images \
					%s/save/images/nii"%source) 
	else:
		os.system("rsync -rvu %s %s zhenzhou@136.187.97.216:~/"%(exclude, source)) 

elif args.biaree:
	if args.image:
		os.system("rsync -rvu hycis@briaree.calculquebec.ca:~/smartNN/save/images \
					%s/save/images/biaree"%source) 		
	else:
		os.system("rsync -rvu %s %s hycis@briaree.calculquebec.ca:~/"%(exclude, source))

elif args.udemrd:
    os.system("rsync -rvu %s /Volumes/Storage/VCTK/Research-Demo \
                wuzhen@elisa2.iro.umontreal.ca:/data/lisa/exp/wuzhen/nii/VoiceCloneCommercial2/"%exclude)

else:
	raise ValueError('options is neither --udem | --nii | --biaree | --udemrd | --image')

