import numpy
import os
import utils
import numpy as np
from scipy.signal import butter,lfilter

def get_sample(train_dir,folder_name,samp_rate):
    video_path = train_dir+"/"+folder_name
    audios=[]
    for file_name in os.listdir(video_path):
        audio = utils.read_audio(os.path.join(video_path,file_name),sr=samp_rate)
        length=len(audio)
        audio = numpy.reshape(audio,(length))
        audios.append(audio)
    audios_numpy = numpy.array(audios,dtype=object)
    return audios_numpy

def pre_fun(sample):
    lenth = len(sample)
    x = numpy.array(sample)
    for i in range(1,int(lenth),1):
        x[i] = x[i]-0.95*x[i-1]
    return x

def butter_lowpass(cutoff,f,order=5):
    b,a = butter(order,cutoff,btype='lowpass',analog=False,fs=f)
    return b,a

def lowpass_filter(sample,cutoff,f,order=5):
    b,a = butter_lowpass(cutoff,f,order=order)
    result = lfilter(b,a,sample)
    return result

def getenergy(sample):
    s1 = sample*sample
    energy = np.mean(s1)
    return energy

def vad(sample,sample_rate,hlen):
    
    #step = int(hlen*sample_rate)
    step = 100
    length = len(sample)
    mean_energy = getenergy(sample)
    flag = 0
    for i in range(0,length,step):
        energy = getenergy(sample[i:i+step])
        if energy >= 0.01*mean_energy:
            if flag == 0:
                result = sample[i:i+step]
                flag = 1
            else:
                result = np.hstack([result,sample[i:i+step]])
    
    return result