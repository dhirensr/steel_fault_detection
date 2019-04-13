
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init()


df = pd.read_csv(os.getcwd()+'/steel_data.csv')

#df=h2o.import_file(os.getcwd()+'/steel_data.csv')
print type(df)
df.loc[df['V28']==1,'Output']=1
df.loc[df['V29']==1,'Output']=2
df.loc[df['V30']==1,'Output']=3
df.loc[df['V31']==1,'Output']=4
df.loc[df['V32']==1,'Output']=5
df.loc[df['V33']==1,'Output']=6
df.loc[df['Class']==2,'Output']=7


df=h2o.H2OFrame(df)
#print df
f=['V1','V2','V3','V4','V5','V6','V7']
train,test  = df.split_frame([0.51])
#train,test= df.split_frame([0.51])
X_train=train.col_names[:-8]
print X_train
Y_train=train.col_names[-1]	



def RandomForest(estimators=10):
	model= H2ORandomForestEstimator(model_id="rf_steel_plates"+str(random.randint(1,10000)),ntrees=200,stopping_rounds=2,score_each_iteration=True,seed=1000000)
	model.train(X_train,Y_train, training_frame=df)
	path=h2o.save_model(model,path=os.getcwd())
	result=model.predict(test[:-8])
	print(model.model_performance(test))
RandomForest()

#h2o.cluster().shutdown()