import feature
import signalprocess
import config
import numpy as np
import pandas
from sklearn.mixture import GaussianMixture as GMM
def main():
    samp_rate=config.Audio.samp_rate
    flen=config.Audio.flen
    hlen=config.Audio.hlen
    datapath=config.task2_datapath
    import joblib

    train_folders=config.getdir(datapath)
    train_folders.sort(key=lambda x:int(x[2:]))

    flag=False
    labels=[]
    j = 1
    for folder in train_folders:

        sample_arrays=signalprocess.get_sample(datapath,folder,16000)
        for sample_array in sample_arrays:
            #sample_array = signalprocess.pre_fun(sample_array)
            mfccs=feature.get_mfcc(sample_array,16000,flen,hlen)
            if not flag:
                dataset_numpy=np.array(mfccs)
                flag=True
            elif flag:
                dataset_numpy=np.vstack((dataset_numpy,mfccs))
            #dataset_pandas=pandas.DataFrame(dataset_numpy)
            for i in range(0,mfccs.shape[0]):
                labels.append(folder)
        flag = False        
        gmm = GMM(n_components=16,max_iter=200,covariance_type='diag',n_init=3)
        gmm.fit(dataset_numpy)
        joblib.dump(gmm,"task_2/model/model_"+str(j)+".plk")
        j = j+1
   #dataset_pandas["speaker"]=labels
   #dataset_pandas.to_csv("dataset.csv",index=False,encoding="gbk")


if __name__=="__main__":
    main()
