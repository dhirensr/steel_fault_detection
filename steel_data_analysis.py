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



df_class1 = df[df.Class==1]
df_class2 = df[df.Class==2]
df_class2_upsampled = sk.utils.resample(df_class2,
                                        replace=True,
                                        n_samples=1268)
df = pd.concat([df_class1, df_class2_upsampled])
print(len(df))
sns.countplot(x="Class", data= df)

df.loc[df['Class'] == 1, 'V34'] = 0
df.loc[df['Class'] == 2, 'V34'] = 1


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-8], df[['V28','V29','V30','V31','V32','V33','V34']] ,test_size=0.3)



def RandomForest(estimators=10):
    clf = RandomForestClassifier(n_estimators = estimators)
    clf.fit(X_train, y_train)

    #clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X_train,y_train)
    y2_predicted=clf.predict(X_test)
    #print(y2_predicted,y_test)
    print("Accuracy_Score=",sk.metrics.accuracy_score(y_test,y2_predicted))
    #print("Feature Importances",clf.feature_importances_)
    imp_features=clf.feature_importances_
    imp_features=imp_features[:27]
    imp_feature_names=X_train.columns[:27]

    y_pos=np.arange(len(imp_feature_names))
    return sk.metrics.accuracy_score(y_test,y2_predicted)

    #plt.bar(x=y_pos,height=imp_features,width=0.8)
    #plt.xticks(y_pos,imp_feature_names)


def Kneighbors(k=10):
    clf= KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    y2_predicted=clf.predict(X_test)
    print("Accuracy_Score=",sk.metrics.accuracy_score(y_test,y2_predicted))
    return sk.metrics.accuracy_score(y_test,y2_predicted)
