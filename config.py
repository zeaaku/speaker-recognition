import numpy as np
import os
task2_datapath = "./train"
class Audio:
    samp_rate = 16000
    fsize = 512
    flen = fsize/samp_rate
    hlen = 0.02
class Audio_1:
    samp_rate = 44100
    fsize = 512
    flen = fsize/samp_rate
    hlen = 0.01
class window:
    hamming = lambda x:0.54-0.46*np.cos((2*np.pi*x)/(Audio.fsize-1))
class Model:
    n_splits = 300
def getdir(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir,name))]
    