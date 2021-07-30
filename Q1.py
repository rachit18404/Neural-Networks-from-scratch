'''!pip install keras
!pip install tensorflow

!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/rapids-colab.sh stable

import sys, os

dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
sys.path
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

!apt install libopenblas-base libomp-dev'''

import numpy as np
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from cuml.manifold import TSNE


class MyNeuralNetwork:
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):

        self.X_prev=[0]*(n_layers-1)  
        self.y_prev=[0]*(n_layers-1)

        self.n_layers=n_layers
        self.layer_sizes=layer_sizes
        self.activation=activation       #saving parameters
        self.learning_rate=learning_rate
        self.weight_init=weight_init
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.Layers=[]
        self.costs=[]
        self.test_cost=[]
        self.Bias=[]
        self.Z=[0]*(n_layers-1)
        
        if weight_init== "random":
            for i in range(n_layers-1):                                                      #initialising weights and bias
                self.Layers.append(self.random_init(   (self.layer_sizes[i],self.layer_sizes[i+1])   )   )
                self.Bias.append(np.zeros(self.layer_sizes[i+1]))
            for weight in range(n_layers-1):
                    bi=self.Bias[weight]
                    bi=[[bi[i]] for i in range(len(bi))]
                    self.Bias[weight]=bi

        if weight_init== "zero":
            for i in range(n_layers-1):
                self.Layers.append(self.zero_init(   (self.layer_sizes[i],self.layer_sizes[i+1])   )   )
                self.Bias.append(np.zeros(self.layer_sizes[i+1]))
            for weight in range(n_layers-1):
                    bi=self.Bias[weight]
                    bi=[[bi[i]] for i in range(len(bi))]
                    self.Bias[weight]=bi                

        if weight_init== "normal":
            for i in range(n_layers-1):
                self.Layers.append(self.normal_init(   (self.layer_sizes[i+1],self.layer_sizes[i])   )   )
                self.Bias.append(np.zeros(self.layer_sizes[i+1]))
            for weight in range(n_layers-1):
                    bi=self.Bias[weight]
                    bi=[[bi[i]] for i in range(len(bi))]
                    self.Bias[weight]=bi       



            
    def relu(self, X):
        X[X<=0]=0
                                      #relu
        return X

    def relu_grad(self, X):
        X[X>0]=1                      #reu_grad
        X[X<=0]=0.1

        return X

    def sigmoid(self, X):
        
        ex=np.exp(-1*X)
        den=1+ex                           #sigmoid
        x_calc=np.reciprocal(den)

        return x_calc

    def sigmoid_grad(self, X):

        x_calc=(1-self.sigmoid(X))*self.sigmoid(X)
                                                        #sigmoid_grad
        return x_calc

    def linear(self, X):
        return 1*X                       #linear
        

    def linear_grad(self, X):
      Xx= np.zeros(shape=X.shape)+1          #linear_grad
      return Xx

    def tanh(self, X):
        return np.tanh(X)
                                        #tanh and tanh_grad
    def tanh_grad(self, X):

        return 1 - np.square(np.tanh(X))

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)


    def softmax_grad(self, X):
        return self.softmax(X)/np.sum(self.softmax(X),axis=0) * (1-self.softmax(X)/np.sum(self.softmax(X),axis=0)) #softmax_grad
       

    def zero_init(self, shape):
        return np.zeros([shape[1], shape[0]])      #defining weights
        

        

    def random_init(self, shape):
        return np.random.randn(shape[1], shape[0])*0.01

        

    def normal_init(self, shape):
        
        return (np.random.normal(0,1,shape))*0.01


    def logloss(self,y, y_pred):
        n = y_pred.shape[0]	
        return (-np.sum(y * np.log(y_pred))) / n        #costfunction
        
  

    def d_logloss(self,y, y_pred):
        return y_pred - y   #cost derivative
     

    def feed_forward(self,X_prev,W,b,weight):

        self.Z[weight]=np.dot(W[weight], X_prev[weight]) +b[weight]
                                                                                  #feedforwarding
  
        if weight==3:
          X=self.softmax(self.Z[weight])
        else:

          if self.activation=='relu':
              X=self.relu(self.Z[weight])
          elif self.activation=='sigmoid':
              X=self.sigmoid(self.Z[weight])
          elif self.activation=='linear':
              X=self.linear(self.Z[weight])                #choosing activation
          elif self.activation=='tanh':
              X=self.tanh(self.Z[weight])
          elif self.activation=='softmax':
              X=self.softmax(self.Z[weight])


        
        return X


    def backpropogation(self,dX,X_prev,Layers,Bias,layer):
        #print(self.sigmoid_grad(self.Z[layer]).shape,dX.shape)
        if self.activation=='relu':
          dZ = np.multiply(self.relu_grad(self.Z[layer]) , dX)   
        elif self.activation=='sigmoid':
          dZ = np.multiply(self.sigmoid_grad(self.Z[layer]) , dX)
        elif self.activation=='linear':
          dZ = np.multiply(self.linear_grad(self.Z[layer]) , dX)            #backpropogation using required gradient
        elif self.activation=='tanh':
          dZ = np.multiply(self.tanh_grad(self.Z[layer]) , dX)
        elif self.activation=='softmax':
          dZ = np.multiply(self.softmax_grad(self.Z[layer]) , dX)
        
        dW = 1/dZ.shape[1] * np.dot(dZ, self.X_prev[layer].T)

        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)

        dX_prev = np.dot(self.Layers[layer].T, dZ)
        Layers[layer] = Layers[layer] - self.learning_rate * dW
        Bias[layer] = Bias[layer] - self.learning_rate * db

        return dX_prev
    
        


    def fit(self, X_train, y_train,X_test,y_test):

        XX_train= np.array_split(X_train,int(X_train.shape[0]//self.batch_size))
        yy_train=np.array_split(y_train,int(X_train.shape[0]//self.batch_size))
  
    
        for epochs in range(self.num_epochs):
             print(epochs)    
             for i in range(int(X_train.shape[0]//self.batch_size)):         
                X=XX_train[i].T                           #fitting our data
                y=yy_train[i]

                for weight in range(len(self.Layers)):
                   
                    self.X_prev[weight] = X 
                    X = self.feed_forward(self.X_prev,self.Layers,self.Bias,weight)
                    
                
                if i==int(X_train.shape[0]//self.batch_size)-1:      
                    cost = 1/X.shape[1] * np.sum(self.logloss(y.T, X))                 #storing cost at each iteration for train set
                    print(cost)
                    self.costs.append(cost)
            
                dX = self.d_logloss(y.T, X)
    
                for layer in reversed(range(len(self.Layers))):
                    dX = self.backpropogation(dX,self.X_prev,self.Layers,self.Bias,layer)  


                self.X_prev=[0]*(self.n_layers-1)            
                self.Z=[0]*(self.n_layers-1)

             Xt=X_test.T
            
             yt=y_test
             
             for weight in range(self.n_layers-1):
                    self.y_prev[weight] = Xt
                    Xt = self.feed_forward(self.y_prev,self.Layers,self.Bias,weight)       #storing cost at each iteration for validation set
            
            
             tc = 1/Xt.shape[1] * np.sum(self.logloss(yt.T, Xt))
             self.test_cost.append(tc)
        global weight_cost_relu   
        global bias_relu
        global weight_cost_sigmoid 
        global bias_sigmoid
        global weight_cost_tanh
        global bias_tanh
        global weight_cost_linear   
        global bias__linear
        if self.activation=="relu":
          weight_cost_relu=self.Layers         #saving weight
          bias_relu=self.Bias
        elif self.activation=="sigmoid":
          weight_cost_sigmoid=self.Layers
          bias_sigmoid=self.Bias
        elif self.activation=="tanh":
          weight_cost_tanh=self.Layers
          bias_tanh=self.Bias
        elif self.activation=="linear":
          weight_cost_linear=self.Layers
          bias_linear=self.Bias


    
        plt.plot(range(self.num_epochs),self.costs)
        plt.plot(range(self.num_epochs),self.test_cost)
        plt.show()       #plot
        return self

    def predict_proba(self, X_test):
      y_pred = X_test
      for layer in range(self.n_layers-1):           #probability as output
        self.y_prev[layer] = y_pred  
        y_pred=self.feed_forward(self.y_prev,self.Layers,self.Bias,layer)
      return y_pred.T

    def predict(self, X_test):
        y_pred = X_test
        for layer in range(self.n_layers-1):
          self.y_prev[layer] = y_pred  
          y_pred=self.feed_forward(self.y_prev,self.Layers,self.Bias,layer)
                                                      #predicted result
        y_pred=y_pred.T
        #print(y_pred.shape)
        y_pred=np.argmax(y_pred,axis=1)
        
          
        return y_pred
    

    def score(self, X_test, y_test):
        y_pred = X_test
        for layer in range(self.n_layers-1):
          self.y_prev[layer] = y_pred  
          y_pred=self.feed_forward(self.y_prev,self.Layers,self.Bias,layer)      #Accuracy score

        y_pred=y_pred.T
        y_pred=np.argmax(y_pred,axis=1)
        y_test = np.where(y_test==1)[1]
        return np.count_nonzero(y_pred==y_test)/len(y_pred)
    

    def tSNE(self, X_train):
      y_pred = X_train
      for layer in range(self.n_layers-2):
        self.y_prev[layer] = y_pred                                #dataset for tSNE
        y_pred=self.feed_forward(self.y_prev,self.Layers,self.Bias,layer)
      print(y_pred)
      return y_pred
