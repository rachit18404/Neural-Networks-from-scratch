import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
f=h5py.File("part_A_train.h5","r")

X=f.get("X")              #Reading X and y, converting it to array, dimension of y is reduced to 1 from 10
X=np.asarray(X)
y=f.get("Y")
y=np.asarray(y)                     
y=np.where(y==1)[1]


scaler=StandardScaler()
X,Xtest,y,ytest=train_test_split(X,y,test_size=1/5,random_state=1,stratify=y)   #Standardisation
yy=pd.DataFrame(y)
yytest=pd.DataFrame(ytest)

print(yy[0].value_counts())
print(yytest[0].value_counts())                     #Class frequency on training and testing samples

scaler.fit(X)
X=scaler.transform(X)
Xtest=scaler.transform(Xtest)

def SVDD(X,Xtest,y,ytest):

    svd = TruncatedSVD(n_components=170)       #Redcing dimensions to 170 using truncated svd
    svd.fit(X,y)
    X=svd.transform(X)
    Xtest=svd.transform(Xtest)
    print("n components",170)



    logistic=LogisticRegression(max_iter=15000)
    logistic.fit(X,y)                            #Accuracy score after svd using logistic regression
    print("Accuracy",logistic.score(Xtest,ytest))



    tsne = TSNE(n_components=2)    #t-SNE, dimensions reduced to 2

    X=tsne.fit_transform(X)


    df_new=pd.DataFrame(X[:,0],columns=["tsne1"])
    df_new["tsne2"]=X[:,1]
    df_new["y"]=y
    plt.figure(figsize=(16,10))   #plot for t-SNE between two components
    sns.scatterplot(   x="tsne1", y="tsne2",   hue="y",   palette=sns.color_palette("hls", 10),   data=df_new,   legend="full",   alpha=0.9)
    plt.show()


def PCAA(X,Xtest,y,ytest):

    pca=PCA(0.90)       #Redcing dimensions such that 90% of variations is conserved
    pca.fit(X)
    print("n components",pca.n_components_)
    X=pca.transform(X)
    Xtest=pca.transform(Xtest)


    logistic=LogisticRegression(max_iter=15000)
    logistic.fit(X,y)
    print("Accuracy",logistic.score(Xtest,ytest))   #Accuracy score after svd using logistic regression



    tsne = TSNE(n_components=2)

    X=tsne.fit_transform(X)                  #t-SNE, dimensions reduced to 2


    df_new=pd.DataFrame(X[:,0],columns=["tsne1"])
    df_new["tsne2"]=X[:,1]
    df_new["y"]=y
    plt.figure(figsize=(16,10)) #plot for t-SNE between two components
    sns.scatterplot(   x="tsne1", y="tsne2",   hue="y",   palette=sns.color_palette("hls", 10),   data=df_new,   legend="full",   alpha=0.9)
    plt.show()

SVDD(X,Xtest,y,ytest)
