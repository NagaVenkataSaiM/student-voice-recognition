import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc, delta
 
def extract_features(audio,rate):
  """extract 20 dim mfcc features from audio file, perform CMS and combine
  delta to make 40 dim feature vector"""    
  print(type(audio))
  print("Code by Nvsai")
  mfcc_feature = mfcc(audio, rate, winlen=0.020,preemph=0.95,numcep=20,nfft=1024,ceplifter=15,highfreq=6000,nfilt=55,appendEnergy=False)
  
  # feature scaling
  mfcc_feature = preprocessing.scale(mfcc_feature)
  delta_feature = delta(mfcc_feature, 2) # calculating delta
  # stacking delta features with common features
  combined_features = np.hstack((mfcc_feature, delta_feature))
  return combined_features 
