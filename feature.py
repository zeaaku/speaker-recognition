import python_speech_features as psf
from scipy import signal
import config
import numpy as np
from sklearn import preprocessing
import signalprocess

def get_mfcc(sample,sample_rate,flen,hlen):
    #sample = trainfun.vad(sample,sample_rate,hlen)
    #sample = signalprocess.pre_fun(sample)
    #sample = signalprocess.lowpass_filter(sample,6000,sample_rate,10)
    mfccs = psf.mfcc(sample,sample_rate,winlen=flen,winstep=hlen,numcep=20,winfunc=config.window.hamming,nfft=512)
    mfccs = preprocessing.scale(mfccs)
    delta_1 = psf.base.delta(mfccs,1)
    delta_2 = psf.base.delta(mfccs,2)
    feature = np.hstack((mfccs,delta_1))
    #mfccs = np.array(mfccs)
    mfccs = feature[:,1:]
    return mfccs
