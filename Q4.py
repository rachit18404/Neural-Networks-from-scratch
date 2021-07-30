import h5py
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


f=h5py.File("part_B_train.h5","r")

X=f.get("X")
X=np.asarray(X)


y=f.get("Y")
y=np.asarray(y)
y=np.where(y==1)[1]         
X,Xtest,y,ytest=train_test_split(X,y,test_size=1/5,random_state=1)

pca=PCA(0.90)
pca.fit(X)
X=pca.transform(X)      #Apllying PCA to data
Xtest=pca.transform(Xtest)



unique=np.unique(y)        
listy=y.tolist()

D={}
X=X.tolist()
for i in range(len(X)):             #Grouping the rows by their class in dictionary
        if y[i] not in D:
           
            D[y[i]]=[X[i]]
        else:
            
            D[y[i]].append(X[i])
            

X=np.asarray(X)
for i in D:
    D[i]=np.asarray(D[i])
    


mean={}
std={}
for i in D:
    mean[i]=np.mean(D[i],axis=0)
    std[i]=np.std(D[i],axis=0)             #Mean and std for every feature of every class

ypred=[]
for row in range(Xtest.shape[0]):
    l=[]
    for lbl in unique:
        prob=listy.count(lbl)/len(y)
        for elem in range(Xtest.shape[1]):
            
            
            prob*=stats.norm.pdf(Xtest[row][elem],mean[lbl][elem],std[lbl][elem])                      #Evaluating using mean and std from train in test set
            
        l.append(prob)
    
    ypred.append(l.index(max(l)))
    

print(np.count_nonzero(ypred==ytest)/len(ypred)*100) 


gnb = GaussianNB()           #Using sklearn
gnb.fit(X, y)
ypred=gnb.predict(Xtest)
print("From sklearn",np.count_nonzero(ypred==ytest)/len(ypred)*100)

    
