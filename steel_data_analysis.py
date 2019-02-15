import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression#CV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import sklearn as sk
import matplotlib.pyplot as plt
import os
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.getcwd()+'/steel_data.csv')

sns.countplot(x="Class", data= df)



#print(len(df))
#sns.countplot(x="Class", data= df)

df.loc[df['Class'] == 1, 'V34'] = 0
df.loc[df['Class'] == 2, 'V34'] = 1
count=0
cl_count_test={'Pastry':count,'Z_Scratch':count, 'K_Scatch':count, 'Stains':count,'Dirtiness':count, 'Bumps':count,'Other_Faults':count}
cl_count_predicted={'Pastry':count,'Z_Scratch':count, 'K_Scatch':count, 'Stains':count,'Dirtiness':count, 'Bumps':count,'Other_Faults':count}
classes={'V28':'Pastry','V29':'Z_Scratch','V30':'K_Scatch','V31':'Stains','V32':'Dirtiness','V33':'Bumps','V34':'Other_Faults'}

def split_data(features=['V1','V2','V3','V4','V5','V6','V7']):

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[['V28','V29','V30','V31','V32','V33','V34']] ,test_size=0.3)
    return X_train,X_test,y_train,y_test


def RandomForest(attributes,estimators=10):
    X_train,X_test,y_train,y_test=split_data(attributes)
    clf = RandomForestClassifier(n_estimators = estimators)
    clf.fit(X_train, y_train)

    #clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X_train,y_train)
    y_predicted=clf.predict(X_test)
    #print(y2_predicted,y_test)
    #print("Accuracy_Score=",sk.metrics.accuracy_score(y_test,y_predicted))
    #print("Feature Importances",clf.feature_importances_)
    imp_features=clf.feature_importances_
    imp_features=imp_features[:27]
    imp_feature_names=X_train.columns[:27]

    y_pos=np.arange(len(imp_feature_names))

    y_predicted_df=pd.DataFrame(y_predicted,columns=y_test.columns)
    cnf_matrix = get_cnf_matrix(y_test,y_predicted_df)
    test_count,predicted_count=get_bar_data(y_test,y_predicted_df)

    return cnf_matrix,sk.metrics.accuracy_score(y_test,y_predicted),test_count,predicted_count

    #plt.bar(x=y_pos,height=imp_features,width=0.8)
    #plt.xticks(y_pos,imp_feature_names)


def Kneighbors(attributes,k=10):
    X_train,X_test,y_train,y_test=split_data(attributes)
    clf= KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    y_predicted=clf.predict(X_test)
    #print("Accuracy_Score=",sk.metrics.accuracy_score(y_test,y_predicted))
    y_predicted_df=pd.DataFrame(y_predicted,columns=y_test.columns)
    cnf_matrix = get_cnf_matrix(y_test,y_predicted_df)
    test_count,predicted_count=get_bar_data(y_test,y_predicted_df)
    return cnf_matrix,sk.metrics.accuracy_score(y_test,y_predicted),test_count,predicted_count

def get_cnf_matrix(y_test,y_predicted_df):
    cnf_matrix = sk.metrics.confusion_matrix(y_test.values.argmax(axis=1), y_predicted_df.values.argmax(axis=1))
    return cnf_matrix

def get_bar_data(y_test,y_predicted_df):
    count=0
    for i in range(28,35):
        index='V'+str(i)
        test_class=(y_test[index]==1).sum() #y_test[y_test[[index]]==1].sum()

        cl_count_test[classes[index]]= test_class

        predicted_class=(y_predicted_df[index]==1).sum()
        cl_count_predicted[classes[index]]= predicted_class


    return list(cl_count_test.values()),list(cl_count_predicted.values())
