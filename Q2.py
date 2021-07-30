import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
train_df=pd.read_csv("weight-height.csv")

X=train_df["Height"]
y=train_df["Weight"].values     
X=np.asarray(X)

y=np.asarray(y)

X,Xtest,y,ytest=train_test_split(X,y,test_size=1/5,random_state=1)
def Bootstraping():
    


    totalsamples=[]
    alltestsamples=[]
    for i in range(1500):
        l=[]                                      #Making 1500 dataset of 8000 samples
        ltest=[]
        RandomI_ListOfIntegers = [random.randrange(1, 8000) for i in range(8000)]               #Generating random 8000 samples
        for j in range(len(RandomI_ListOfIntegers)):
            l.append([X[RandomI_ListOfIntegers[j]]])
            ltest.append(y[RandomI_ListOfIntegers[j]])

        
        l=np.asarray(l)
        ltest=np.asarray(ltest)

        lr=LinearRegression()
        lr.fit(l,ltest)                                               
        alltestsamples.append(lr.predict(Xtest.reshape(-1,1)))        #Predicting on testset for all the training set
            
    alltestsamples=np.asarray(alltestsamples)
    means=(np.mean(alltestsamples,axis=0))         #Taking mean for every feature predicted value



    bias=means-ytest        #calculating bias

    variance=np.var(alltestsamples,axis=0) #calculating variance

    MSE=bias**2

    MSE=np.mean(MSE)
    bias=bias.mean()

    print("bias",bias)
    variance=np.mean(variance)
    print("variance",variance)


    print("MSE",MSE)             #Calculating MSE

    print(MSE-bias**2- variance)
Bootstraping()
