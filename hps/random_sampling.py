# -*- coding: utf-8 -*-
import argparse
import multiprocessing
import numpy
import numpy.random as rng
import os
import socket
import sys
import time
from jobman import DD, flatten

sys.path.insert(0, os.getcwd())
from model_config import model_config





def worker(num, cmd):
    """worker function"""
    print 'Worker %s' %num
    os.system(cmd)
    return


def exp_sampling(((low,high),t)):
    low = numpy.log(low)
    high = numpy.log(high)
    return t(numpy.exp(rng.uniform(low,high)))


def cmd_line_embed(cmd, config):
    for key in config:
    #print 'config====', config
        #import pdb
        #pdb.set_trace()
        if type(config[key])==type(()):
            type_val = config[key][1]
            if type_val == int:
		#print '============', config[key], key
                min_val, max_val = config[key][0]
                val = rng.randint(min_val, max_val)
            elif type_val == float:
                val = exp_sampling(config[key])
            else:
                raise NotImplementedError('type %s not supported!'%type_val)
            cmd += key + '=' + `val` + ' '
        elif type(config[key])==type([]):
            v = str(config[key]).replace(' ', '')
            cmd += key + '=' + str(v) + ' '
        else:
            cmd += key + '=' + `config[key]` + ' '
    return cmd


def get_cmd(model, mem, use_gpu, host):
    cmd = 'jobdispatch --file=commands.txt --exp_dir=%s --mem=%s'%(model, mem)
    if 'umontreal' in host:
        # Lisa cluster.
        cmd += ' --condor '
    elif 'ip05' in host:
        # Mammouth cluster.
        cmd += ' --bqtools '
    elif 'briaree' in host:
        # Briaree cluster.
        if use_gpu:
            cmd += ' --torque --gpu --env=THEANO_FLAGS=device=gpu '
        else:
            cmd += ' --torque --env=THEANO_FLAGS=floatX=float32 '
    else:
        host = 'local'
#     if use_gpu:
#         assert 'ip05' not in host
#         if 'briaree' in host:
#             cmd += ' --torque --gpu --env=THEANO_FLAGS=device=gpu '
    return cmd


if __name__=='__main__':
    #path = '/data/lisa/exp/wuzhen/code/'
    #sys.path.append(path)
    #os.environ['PYTHONPATH'] += ':' + path
    #print 'PYTHONPATH:', os.environ['PYTHONPATH']
    
    parser = argparse.ArgumentParser(description='''Train mlps by launching
        jobs on clusters or locally.''')

    parser.add_argument('-g', '--use_gpu', action='store_true',
                        help='''Models will be trained with gpus''')

    parser.add_argument('-n', '--total_number_jobs', type=int, dest='n_jobs',
                        default=1, help='''The total number of jobs that will
                                             be launched.''')

    parser.add_argument('-m', '--memory', type=int, dest='mem',
                        default=2000, 
                        help='''Memory usage limit by job in MB.''')

    parser.add_argument('-c', '--number_concurrent_jobs', type=int,
                        dest='n_concur_jobs',
                        help='''If this option is used, then jobs will be
                                launched locally and it specifies the
                                number of concurrent jobs that can
                                running at the same time at most.''')
    
    parser.add_argument('-r', '--record', action='store_true',
                       help='''If this option is used, then the outputs from
                               terminal will be saved into file''')
    # TODO: ajouter assert pour s'assurer que lorsqu'on lance des jobs avec gpu, seulement
    # 1 job puisse etre lance localement.
    args = parser.parse_args()
    print args
    cmds = []
    exps_by_model = {}

    ######### MODEL #########
    model = 'autoencoder'
    jobs_folder = 'jobs'
    #########################

    host = socket.gethostname()
    print 'Host = ', host
    # TODO: Hardcoded model name.

    if args.n_concur_jobs:
        host = 'local'
    cmd = get_cmd(model, args.mem, args.use_gpu, host)
    if not os.path.exists(jobs_folder):
        os.mkdir(jobs_folder)
    f = open('jobs/commands.txt','w')
    #import pdb
    #pdb.set_trace()
    print 'commands'

    for i in range(args.n_jobs):
        # TODO: do not hardcode the common options!
        #exp_cmd = 'jobman -r cmdline mlp_training_framework.model.experimTHEANO_FLAGS=profile=True,floatX=float32ent '
        if args.record:
            print('..outputs of job (' + str(i) + ') will be recorded')
            exp_cmd = 'jobman -r cmdline experiment.experiment '
        else:
            exp_cmd = 'jobman cmdline experiment.experiment '
        
        print exp_cmd

        if 'ip05' in host:
            exp_cmd = 'THEANO_FLAGS=floatX=float32 ' + exp_cmd
        if args.use_gpu and host == 'local':
            exp_cmd = 'THEANO_FLAGS=device=gpu ' + exp_cmd
#         import pdb
#         pdb.set_trace()
        exp_cmd = cmd_line_embed(exp_cmd, flatten(model_config[model]))
        f.write(exp_cmd+'\n')
        exps_by_model.setdefault(model, [])
        exps_by_model[model].append(exp_cmd)

    f.close()
#     import pdb
#     pdb.set_trace()
    #print '====cmd====', cmd
    os.chdir(jobs_folder)
    #print '==args.n_concur_jobs', args.n_concur_jobs
    
    import pdb
    pdb.set_trace()
    
    if not args.n_concur_jobs:
        os.system(cmd)
    else:
        print 'Jobs will be run locally.'
        print '%s jobs will be run simultaneously.'%args.n_concur_jobs
        n_jobs = 0
        n_job_simult = 0
        jobs = []
        commands = exps_by_model[model]
        #print '====commands====', commands
        
        import pdb
        pdb.set_trace()

        for command in commands:
            if n_job_simult < args.n_concur_jobs:
                assert len(jobs) <= args.n_concur_jobs
                print command
                p = multiprocessing.Process(target=worker, args=(n_jobs, command))
                jobs.append((n_jobs, p))
                p.start()
                n_jobs += 1
                n_job_simult += 1
                
                #import pdb; pdb.set_trace()
            else:
                ready_for_more = False
                while not ready_for_more:
                    for j_i, j in enumerate(jobs):
                        if 'stopped' in str(j[1]):
                            print 'Job %s finished' %j[0]
                            jobs.pop(j_i)
                            #import pdb
                            #pdb.set_trace()
                            n_job_simult -= 1
                            ready_for_more = True
                            break

        more_jobs = True
        while more_jobs:
            for j_i, j in enumerate(jobs):
                if 'stopped' in str(j[1]):
                    print 'Job %s finished' %j[0]
                    jobs.pop(j_i)
                if len(jobs) == 0:
                    more_jobs = False
                    break
        print 'All jobs finished running.'

