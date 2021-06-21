#!/usr/bin/env python
##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################

import os
from load_data import load_data
import numpy as np
from sklearn.cluster import KMeans


training_data_path = os.environ['TRAIN_DATA_PATH']  # retrieve environmental variable from .env file
mydata = load_data(training_data_path)  # load the data as a numpy array format
fdata=[]
for i in range(mydata.shape[0]):
    fdata.append(mydata[i][8:23])
fdata = np.array(mydata)

class GraspClustering:

     def train(self):
          self.model = KMeans(n_clusters = 8, random_state =0).fit(fdata)
          
     def predict(self,test_data):
          
          final = self.model.predict(test_data)
          return final
    
