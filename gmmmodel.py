from extractfeatures import extract_features
import noisereduce as nr 
import os
from scipy.io import wavfile
import sys
from sklearn import mixture
from pathlib import Path

path="/home/nvsai/programming/python/student-voice-recognition/voice_files"
# filelist = []
# filenamelist=[]
filelist={}
modelslist={}
print("Code by Nvsai")

for root, dirs, files in os.walk(path):
	for file in files:
            filename=file.split(".",1)[0]
            filelist[filename]=os.path.join(root,file)
	    # filelist.append(os.path.join(root,file))
	    # filenamelist.append(file)

print(filelist.keys())
def gmmmodels(file):
    sample_rate, data = wavfile.read(file)
    data=data[:20000]
    # print(data)
    # extract 40 dimensional MFCC & delta MFCC features
    features = extract_features(data, sample_rate)
    gmm = mixture.GaussianMixture(n_components=16,max_iter=250,covariance_type='diag',n_init=1, init_params='random')
    k=gmm.fit(features)  # gmm training
    # print(features)
    prediction=gmm.score(features)
    # print(prediction)
    return k

for filename,filepath in filelist.items():
        modelslist[filename]=gmmmodels(filepath)
        


while True:
        print("Enter 1 to exit or enter file name to predict")
        m=input()
        if m=="1":
                exit()
        bestscore=-1000000
        bestfile="None"
        sample_rate, data = wavfile.read("predict_files/"+m)
        data=data[:20000]
        features = extract_features(data, sample_rate)
        for filename,model in modelslist.items():
                print(model.score(features))
                if model.score(features)>bestscore:
                        bestfile=filename
                        bestscore=model.score(features)
        print(bestscore)
        print("Hello",bestfile)


    
