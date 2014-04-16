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
from datetime import datetime


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
        if type(config[key])==type(()):
            type_val = config[key][1]
            if type_val == int:
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


def get_cmd(model, mem, use_gpu, queue, host):
    dt = datetime.now()
    dt = dt.strftime('%Y%m%d_%H%M_%S%f')
    cmd = 'jobdispatch --file=commands.txt --exp_dir=%s_%s'%(model,dt)
    
    if mem:
        cmd += ' --mem=%s '%mem
    
    if queue:
        cmd += ' --queue=%s '%queue
    
    if 'umontreal' in host:
        # Lisa cluster.
        cmd += ' --condor '
        if mem is None:
            cmd += ' --mem=15000 '
    elif 'ip05' in host:
        # Mammouth cluster.
        cmd += ' --bqtools '
    elif 'briaree1' in host:
        # Briaree cluster.
        if use_gpu:
            cmd += ' --gpu --env=THEANO_FLAGS=device=gpu '
        else:
            cmd += ' --env=THEANO_FLAGS=floatX=float32 '
    else:
        host = 'local'
    return cmd


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='''Train mlps by launching
        jobs on clusters or locally.''')

    parser.add_argument('-g', '--use_gpu', action='store_true',
                        help='''Models will be trained with gpus''')
    
    parser.add_argument('-q', '--queue',
                        help='''The queue to insert the jobs''')

    parser.add_argument('-n', '--total_number_jobs', type=int, dest='n_jobs',
                        default=1, help='''The total number of jobs that will
                                             be launched.''')

    parser.add_argument('-m', '--memory', type=int, dest='mem',
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
                               
    parser.add_argument('--model', help='''choose the model AE or AE_Two_Layers to run''')
    
    # TODO: ajouter assert pour s'assurer que lorsqu'on lance des jobs avec gpu, seulement
    # 1 job puisse etre lance localement.
    args = parser.parse_args()
    print args
    cmds = []
    exps_by_model = {}

    ######### MODEL #########
    print('..Model: ' + args.model)
    assert args.model in ['AE', 'AE_Two_Layers']
    model = args.model
    jobs_folder = 'jobs'
    #########################

    exp_model = 'experiment.%s_exp'%model

    host = socket.gethostname()
    print 'Host = ', host
    # TODO: Hardcoded model name.        

    if args.n_concur_jobs:
        host = 'local'
    cmd = get_cmd(model, args.mem, args.use_gpu, args.queue, host)
    if not os.path.exists(jobs_folder):
        os.mkdir(jobs_folder)
    f = open('jobs/commands.txt','w')

    print '..commands: ', cmd

    for i in range(args.n_jobs):
        # TODO: do not hardcode the common options!
        if args.record:
            print('..outputs of job (' + str(i) + ') will be recorded')
            exp_cmd = 'jobman -r cmdline %s '%exp_model 
        else:
            exp_cmd = 'jobman cmdline %s '%exp_model
        
        print exp_cmd

        if 'ip05' in host:
            exp_cmd = 'THEANO_FLAGS=floatX=float32 ' + exp_cmd
        if args.use_gpu and host == 'local':
            exp_cmd = 'THEANO_FLAGS=device=gpu ' + exp_cmd

        exp_cmd = cmd_line_embed(exp_cmd, flatten(model_config[model]))
        f.write(exp_cmd+'\n')
        exps_by_model.setdefault(model, [])
        exps_by_model[model].append(exp_cmd)

    f.close()

    os.chdir(jobs_folder)
    
    print '..commands: ', cmd
    
    if not args.n_concur_jobs:
        os.system(cmd)
    else:
        print 'Jobs will be run locally.'
        print '%s jobs will be run simultaneously.'%args.n_concur_jobs
        n_jobs = 0
        n_job_simult = 0
        jobs = []
        commands = exps_by_model[model]

        for command in commands:
            if n_job_simult < args.n_concur_jobs:
                assert len(jobs) <= args.n_concur_jobs
                print command
                p = multiprocessing.Process(target=worker, args=(n_jobs, command))
                jobs.append((n_jobs, p))
                p.start()
                n_jobs += 1
                n_job_simult += 1
                
            else:
                ready_for_more = False
                while not ready_for_more:
                    for j_i, j in enumerate(jobs):
                        if 'stopped' in str(j[1]):
                            print 'Job %s finished' %j[0]
                            jobs.pop(j_i)
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
