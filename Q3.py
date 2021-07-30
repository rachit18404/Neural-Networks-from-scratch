import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

class Kfold:

    XX=[]
    yy=[]
    XXtest=[]
    yytest=[]
    
    def __init__(self,k,model,train,test):      #Taking model,X,y as parameter
        self.k=k
        self.model=model
        self.train=train
        self.test=test
     

    def proc(self):
        Ntr=np.array_split(self.train, self.k)
        Nte=np.array_split(self.test, self.k)
       
        l=[None]*self.train.shape[1]
        l=np.array(l)
        lt=[None]
        lt=np.array(lt)
        
        

        alltrain=[]
        alltest=[]
        testpred=[]
        testout=[]
        for i in range(self.k):
            for j in range(self.k):
                if j ==i:                             #Spliting data into K folds
                    testpred.append(Ntr[j])
                    testout.append(Nte[j])
                    pass
                else:
                    l=np.row_stack((l,Ntr[j]))
                    lt=np.concatenate([lt,Nte[j]])
                  
                    #print(Ntr[j])                    
                    #print(" ")
            #print("end")
            l=np.delete(l,(0),axis=0)
           
            lt=np.delete(lt,(0))
            
            alltrain.append(l)
            alltest.append(lt)
            l=[None]*len(self.train[0])
            l=np.array(l)
            lt=[None]
            lt=np.array(lt)
        

            
            
       # print(alltrain)
       # print(alltest)
       # print(testpred)
       # print(testout)
        self.XX=alltrain
        self.yy=alltest
        self.XXtest=testpred
        self.yytest=testout
        
        

    
    def eval(self):
        score=[]
        trainacc=[]
            
        for i in range(self.k):
                self.yy[i]=self.yy[i].astype('int64')
                self.model.fit(self.XX[i],self.yy[i])
                yyout=self.model.predict(self.XXtest[i])
                xxout=self.model.predict(self.XX[i])                           #Performing K fold by taking k-1 folds as train set and  1 fold as test set, and taking mean of them
                trainacc.append(np.count_nonzero(xxout==self.yy[i])/len(xxout))
                score.append(np.count_nonzero(yyout==self.yytest[i])/len(yyout))
        
        return(sum(trainacc)/len(trainacc),sum(score)/len(score))



class Grid_SearchCV:
    trainA=[]
    testA=[]
    bestdepth=0
    
    def __init__(self,model,param_grid,cv):
        self.model=model
        self.param_grid=param_grid                #Taking model and optimal depth dict as parameters 
        self.cv=cv
    
    def fit(self,X,y):
        dlist=self.param_grid["max_depth"]
        for i in range(len(dlist)):                  
            self.model.max_depth=dlist[i]
            K=Kfold(self.cv,self.model,X,y)        #Iterating towards every depth in param_grid and performing K folds cross evaluation on every one of them
            K.proc()
            trainacc,testacc=K.eval()
            self.trainA.append(trainacc)
            self.testA.append(testacc)
            
            
        max_depthaccu=max(self.testA)         
        index=self.testA.index(max_depthaccu)    #setting best depth as the depth where depth with mean of Kfold is maximum for testing set
        self.bestdepth=dlist[index]
        
       
        plt.plot(dlist,self.trainA)           #Accuracy plot
        plt.plot(dlist,self.testA)        
        plt.show()
        
            
            


f=h5py.File("part_B_train.h5","r")
X=f.get("X")
X=np.asarray(X)
y=f.get("Y")                  #Reading file and converting to numpy
y=np.asarray(y)
y=np.where(y==1)[1]





train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=1/5,random_state=0)     # splitting into 80-20

tree_model = DecisionTreeClassifier(max_depth=8,random_state=0)     #Initialising the model






'''GaussianNB'''

def Gauss():
    GN=GaussianNB()
    K=Kfold(4,GN,train_x,train_y)
    K.proc()             #Kfold on Naive Bayes
    print(K.eval())










def EvaluationMetric():
    tree_model = DecisionTreeClassifier(max_depth=4,random_state=0)
    tree_model.fit(train_x,train_y)
    yyout=tree_model.predict(test_x)
    ypred_prob=tree_model.predict_proba(test_x)

    
    '''Confusion matrix'''
    
    unique=np.unique(y)
    CM=[[0]*len(unique) for i in range(len(unique))]
    for i in range(len(test_y)):
            CM[yyout[i]][test_y[i]]+=1
    CM=np.asarray(CM)
   
    print('')

    '''Accuracy'''
    
    acc=np.trace(CM)/np.sum(CM)
    print('')

    '''recall'''
    
    recall=[]
    for i in range(len(unique)):
        total=0
        tp=0
        for j in range(len(unique)):
            total+=CM[j][i]
            if i==j:
                tp+=CM[j][i]
        recall.append(tp/total)
    print('')

    
    '''precision'''
    
    precision=[]
    for i in range(len(unique)):
        total=0
        tp=0
        for j in range(len(unique)):
            total+=CM[i][j]
            if i==j:
                tp+=CM[i][j]
        precision.append(tp/total)
    print('')
    '''F1 Score'''
    
    F1=[]

    for i in range(len(unique)):
        f1=2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        F1.append(f1)
        
    recall=sum(recall)/len(recall)
    precision=sum(precision)/len(precision)
    print("Confusion Matrix")
    print(CM)
    print("Accuracy",acc)
    print("Recall,",recall)
    print("Precision",precision)
    print("F1 Score",sum(F1)/len(F1))
    print('')
    if len(unique)>2:
        print("Micro")
        print("Accuracy for micro",np.trace(CM)/np.sum(CM))
        print("Recall for micro",np.trace(CM)/np.sum(CM))
        print("Precision for micro",np.trace(CM)/np.sum(CM))
        print("F1 Score micro",np.trace(CM)/np.sum(CM))


    '''ROC Curve'''
    
    threshold = [0,0.1,0.2,0.4,0.6,0.8,1]
    if len(unique)>2:
        
        for k in range(ypred_prob.shape[1]):
            tpr=[]
            fpr=[]
            for i in range(len(threshold)):
                TP=0
                FP=0
                FN=0
                TN=0
                for j in range(len(ypred_prob)):
                    if ypred_prob[j][k]>threshold[i]:
                        if test_y[j]==k:
                            TP+=1
                        else:
                            FP+=1
                    else:
                            if(test_y[j]==k):
                                FN+=1
                            else :
                                TN+=1
                
                fpr.append(FP/float(TN+FP))
                tpr.append(TP/float(TP+FN))

            plt.plot(fpr, tpr,marker="o")
            
        plt.plot([0,1],[0,1])
        plt.show()


    else:
        
        tpr=[]
        fpr=[]
        for i in range(len(threshold)):
            TP=0
            FP=0
            FN=0
            TN=0
            for j in range(len(ypred_prob)):
                if ypred_prob[j][1]>threshold[i]:
                    if test_y[j]==1:
                        TP+=1
                    else:
                        FP+=1
                else:
                        if(test_y[j]==1):
                            FN+=1
                        else :
                            TN+=1
            
            fpr.append(FP/float(TN+FP))
            tpr.append(TP/float(TP+FN))

        plt.plot(fpr, tpr,marker="o")
        plt.plot([0,1],[0,1])
        plt.show()


EvaluationMetric()





def savemodel(modelx,filename):
    pickle.dump(modelx, open(filename, 'wb'))   #Saving the model 
 

def loadresult(filename,test_x,test_y):
    loadmodel = pickle.load(open(filename, 'rb'))  #Loading the model and accuracy score
    result = loadmodel.score(test_x, test_y)
    print(result)






param={"max_depth":list(range(1,30))}

grid=Grid_SearchCV(tree_model,param_grid=param,cv=4)          #Performing grid search with cv=4, for 60-20-20 split
grid.fit(train_x,train_y)                         

print("optimal depth",grid.bestdepth)
tree_model.max_depth=grid.bestdepth
tree_model.fit(train_x,train_y)     
savemodel(tree_model,"abc.txt") #saving the model with best optimal parameters
#loadresult("abc",test_x,test_y) 

yyout=tree_model.predict(test_x)

print(np.count_nonzero(yyout==test_y)/len(yyout))  #Accuracy Score




