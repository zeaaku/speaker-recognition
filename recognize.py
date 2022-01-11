import joblib
import feature
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture
import signalprocess
import utils
def findmost(label):
    result = Counter(label)
    most = result.most_common(1)
    most = most[0]
    return most[0]

def findspeaker(sample,sample_rate,flen,hlen):
    #sample = signalprocess.pre_fun(sample)
    sample = signalprocess.vad(sample,sample_rate,hlen)
    mfccs = feature.get_mfcc(sample,sample_rate,flen,hlen)
    like = np.zeros(20)
    for j in range(1,21):
        gmm = joblib.load("task_2/model/model_"+str(j)+".plk")
        scores = np.array(gmm.score(mfccs))
        like[j-1] = scores.sum()
    result = np.argmax(like)+1
    result = utils.ID_dict[result]
    #model = joblib.load("model_mlp.plk")
    #prelabel = model.predict(mfccs)
    #result = findmost(prelabel)
    #print(result)
    return result
