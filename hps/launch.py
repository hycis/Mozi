# -*- coding: utf-8 -*-
import argparse
import multiprocessing
import numpy
import numpy.random as rng
import os
import socket
import sys
import time
import importlib
from jobman import DD, flatten
from datetime import datetime


def worker(num, cmd):
    """worker function"""
    print 'Worker %s' %num
    os.system(cmd)
    return


def exp_sampling((ls,t)):
    assert len(ls) == 2, 'size of list has to be 2'
    low = ls[0]
    high = ls[1]
    low = numpy.log(low)
    high = numpy.log(high)
    return t(numpy.exp(rng.uniform(low,high)))

def cmd_line_embed(cmd, config):
    for key in config:
        if type(config[key])==type(()):
            if config[key][1] in [int, float]:
                val = exp_sampling(config[key])
                cmd += key + '=' + `val` + ' '

            else:
                list_size = len(config[key])
                assert list_size > 0, 'list is empty'
                val = config[key][rng.randint(list_size)]
                cmd += key + '=' + `val` + ' '

        elif type(config[key])==type([]):
            v = str(config[key]).replace(' ', '')
            cmd += key + '=' + str(v) + ' '

        else:
            cmd += key + '=' + `config[key]` + ' '
    return cmd


def get_cmd(model, mem, use_gpu, queue, host, duree, ppn, nb_proc, pmem, gpus, proj):
    dt = datetime.now()
    dt = dt.strftime('%Y%m%d_%H%M_%S%f')
    cmd = 'jobdispatch --file=commands.txt --exp_dir=%s_%s'%(model, dt)

    if nb_proc:
        cmd += ' --nb_proc=%s '%nb_proc

    if ppn:
        cmd += ' --cpu=%s '%ppn

    if mem:
        cmd += ' --mem=%s '%mem

    if use_gpu:
        cmd += ' --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32 '


    if queue:
        cmd += ' --queue=%s '%queue

        # k20 node is in guillimin
        if queue in 'k20':
            if use_gpu:
                # pmem is memory per core, mem is total memory for a job
                cmd += ' --extra_param=:gpus=%s,pmem=%s '%(gpus,pmem)
        # phi is gpu node in guillimin
        elif queue in 'phi':
            if use_gpu:
                cmd += ' --extra_param=:mics=%s,pmem=%s '%(gpus,pmem)

        if queue in 'aw':
            if use_gpu:
                cmd += ' --extra_param=:gpus=%s,pmem=%s '%(gpus,pmem)

        # if queue in 'gpu_4':
        #     if use_gpu:
        #         cmd += ' --extra_param=:nodes=1,gpus=4 '

    if duree:
        cmd += ' --duree=%s '%duree

    if proj:
        cmd += ' --project=%s '%proj

    if 'umontreal' in host:
        # Lisa cluster.
        cmd += ' --condor '
        if mem is None:
            cmd += ' --mem=15000 '
    elif 'ip05' in host:
        # Mammouth cluster.
        cmd += ' --bqtools '
    elif host[:7]  == 'briaree':
        # Briaree cluster.
        if not use_gpu:
            cmd += ' --env=THEANO_FLAGS=floatX=float32 '
        if not proj:
            cmd += ' --project=jvb-000-ae '
    elif host[:6] == 'helios':
        if gpus:
            cmd += ' --extra_param=:gpus=%s '%gpus
        if not proj:
            cmd += ' --project=jvb-000-aa '

    else:
        host = 'local'
    return cmd


def generate_jobs_list(n_jobs, recorded, use_gpu, host, model_config):

    exps_list = []
    for i in range(n_jobs):
        # TODO: do not hardcode the common options!
        if recorded:
            print('..outputs of job (' + str(i) + ') will be recorded')
            exp_cmd = 'jobman -r cmdline experiment.job '
        else:
            exp_cmd = 'jobman cmdline experiment.job '

        if use_gpu and host is 'local':
            exp_cmd = 'THEANO_FLAGS=device=gpu,floatX=float32 ' + exp_cmd

        exp_cmd = cmd_line_embed(exp_cmd, flatten(model_config))
        # print exp_cmd

        exps_list.append(exp_cmd)

    return exps_list


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='''Train mlps by launching
        jobs on clusters or locally.''')

    parser.add_argument('-g', '--use_gpu', action='store_true',
                        help='''Models will be trained with gpus''')

    parser.add_argument('-q', '--queue',
                        help='''The queue to insert the jobs''')

    parser.add_argument('-n', '--total_number_jobs', type=int, dest='n_jobs',
                        help='''The total number of jobs that will be launched on machines.''')

    parser.add_argument('-m', '--mem', default='8000m',
                        help='''Memory usage limit by job in MB.''')

    parser.add_argument('-c', '--n_concur_jobs', type=int,
                        help='''If this option is used, then jobs will be
                                launched locally and it specifies the
                                number of concurrent jobs that can
                                running at the same time at most.''')

    parser.add_argument('-r', '--record', action='store_true',
                       help='''If this option is used, then the outputs from
                               terminal will be saved into file''')

    parser.add_argument('--duree', default='12:00:00', help='''Walltime hh:mm:ss''')

    parser.add_argument('--model', help='''choose the model AE or AE_Two_Layers to run''')

    parser.add_argument('--ppn', help='''indicate the ppn''')

    parser.add_argument('--nb_proc', help='''number of jobs running at any one time''')

    parser.add_argument('--pmem', default='8000m', help='''memory allocation per core''')

    parser.add_argument('--gpus', default='2', help='''memory allocation per core''')

    parser.add_argument('--project', help='''project to which the job is assigned''')

    parser.add_argument('--wait_between_jobs', type=int, default=100, help='''time to wait before launching another new job''')

    # TODO: ajouter assert pour s'assurer que lorsqu'on lance des jobs avec gpu, seulement
    # 1 job puisse etre lance localement.
    args = parser.parse_args()
    print args

    model_config = importlib.import_module("hps.model_configs.%s"%args.model).config

    ######### MODEL #########
    print('..Model: ' + args.model)
    model = args.model

    #### Local Host Name ####
    host = socket.gethostname()
    print '..Host: ', host

    # cwd = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    ### Jobs Folder ###
    jobs_folder = cwd + '/jobs'
    print '..jobs folder:', jobs_folder
    if not os.path.exists(jobs_folder):
        os.mkdir(jobs_folder)
    f = open('{}/commands.txt'.format(jobs_folder),'wb')

    os.chdir(jobs_folder)

    assert args.n_concur_jobs or args.n_jobs, "need to specify whether jobs are gonna run locally or on the cluster"

    if args.n_concur_jobs:
        host = 'local'

    if args.n_jobs:
        print '..Jobs will be dispatched to cluster'

        ## Jobdispatch Command ##
        cmd = get_cmd(model, args.mem, args.use_gpu, args.queue, host,
                    args.duree, args.ppn, args.nb_proc, args.pmem, args.gpus, args.project)
        print '..jobdispatch command: ', cmd

        exps_list = generate_jobs_list(args.n_jobs, args.record, args.use_gpu, host, model_config)
        for exp_cmd in exps_list:
            f.write(exp_cmd + '\n')
        f.close()
        os.system(cmd)


    elif args.n_concur_jobs:
        print '..Jobs will be run locally.'
        print '..%s jobs will be run simultaneously.'%args.n_concur_jobs
        n_jobs = 0
        n_job_simult = 0
        jobs = []
        exps_list = generate_jobs_list(args.n_concur_jobs, args.record, args.use_gpu, host, model_config)
        for exp_cmd in exps_list:
            if n_job_simult < args.n_concur_jobs:
                assert len(jobs) <= args.n_concur_jobs
                print exp_cmd
                p = multiprocessing.Process(target=worker, args=(n_jobs, exp_cmd))
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
            print '..waiting for %s seconds between jobs'%args.wait_between_jobs
            time.sleep(args.wait_between_jobs)

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
