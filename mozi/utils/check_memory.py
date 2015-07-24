import os
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return virtual memory usage in bytes.
    '''
    return _VmB('VmSize:') - since

def peak_Vm(since=0.0):
    '''Return the peak virtual memory usage in bytes
    '''
    return _VmB('VmPeak:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since

def peak_resident(since=0.0):
    '''Return the peak resident memory usage in bytes.
    '''
    return _VmB('VmHWM:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since


def get_mem_usage():
    rstr = 'virtual memory: ' + str(memory()/1024/1024) + ' MB\n'
    rstr += 'peak virtual memory: ' + str(peak_Vm()/1024/1024) + ' MB\n'
    rstr += 'resident memory: ' + str(resident()/1024/1024) + ' MB\n'
    rstr += 'peak resident memory: ' + str(peak_resident()/1024/1024) + ' MB\n'
    rstr += 'stacksize: ' + str(resident()/1024/1024) + ' MB\n'

    return rstr
