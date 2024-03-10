#Melih Şahin ENGR 421 HW-3

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X = np.genfromtxt("hw03_data_points.csv", delimiter = ",")
y = np.genfromtxt("hw03_class_labels.csv", delimiter = ",").astype(int)



i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(1 - i1, cmap = "gray")
plt.show()
fig.savefig("hw03_images.pdf", bbox_inches = "tight")



# STEP 3
# first 60000 data points should be included to train
# remaining 10000 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train=X[0:60000]
 
    y_train=y[0:60000]
 
 
    X_test=X[60000: 70000]
 
    y_test=y[60000:70000]

    
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def sigmoid(X, W, w0):
    # your implementation starts below

    aa=len(X)
    bb=len(X[0])



    h=np.matmul(X,W)
    h+=w0
    h=np.exp(-h)
    h+=1
    h=1/h
    
    scores=h
    
    
    
            
            

    
    # your implementation ends above
    return(scores)
            
            

    
    # your implementation ends above
    return(scores)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def one_hot_encoding(y):
    # your implementation starts below
    Y=np.zeros((len(y),10))
    for a in range(y.size):
        for b in range(10):
            if y[a]==b:
                Y[a][b]=1
                
    
    # your implementation ends above
    return(Y)



np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.01, high = 0.01, size = (D, K))
w0_initial = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))



# STEP 6
# assuming that there are D features and K classes
# should return a numpy array with shape (D, K)
def gradient_W(X, Y_truth, Y_predicted):
    # your implementation starts below

    #h=(Y_truth-Y_predicted)*(Y_truth*Y_predicted)*(1/(Y_predicted-1))    
    h=-(Y_truth-Y_predicted)*(Y_predicted)*(1-Y_predicted)
    gradient=np.matmul(X.T,h)
    # your implementation ends above
    return(gradient)

# assuming that there are K classes
# should return a numpy array with shape (1, K)
def gradient_w0(Y_truth, Y_predicted):
    # your implementation starts below
    gradient=np.zeros((1,10))
    h=-(Y_truth-Y_predicted)*(Y_predicted)*(1-Y_predicted)
    
    for a in range(10):

        
        jj=h[:, a]
        gradient[0][a]=sum(jj)
        
   
            
        
        
    
    # your implementation ends above
    return(gradient)



# STEP 7
# assuming that there are N data points and K classes
# should return three numpy arrays with shapes (D, K), (1, K), and (200,)
def discrimination_by_regression(X_train, Y_train,
                                 W_initial, w0_initial):
    eta = 1.0 / X_train.shape[0]
    iteration_count = 200

    W = W_initial
    w0 = w0_initial
        
    # your implementation starts below
    objective_values=[]
    yy=Y_train
    
    

    for a in range(iteration_count):



        h=sigmoid(X_train, W, w0)


        
        delta_w0=gradient_w0(yy, h)
        delta_W=gradient_W(X_train,yy,h)
        def safelog(x):
           return(np.log(x + 1e-100))
       
        
        
        objective_values = np.append(objective_values,-np.sum(Y_train * safelog(h) + (1 - Y_train) * safelog(1 - h)))

        
        W-=eta*delta_W
        w0-=eta*delta_w0
    #w0=  [[-0.06684353, -0.42498154, -0.44906758, -0.23974295, -0.1976174 , -0.34234379 ,-0.17061777, -0.71142747 ,-0.46719659 ,-0.3503129 ]]
    
    
        
        
    
    
    # your implementation ends above
    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)
print(W)
print(w0)
print(objective_values[0:10])



fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("hw03_iterations.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are N data points
# should return a numpy array with shape (N,)
def calculate_predicted_class_labels(X, W, w0):
    # your implementation starts below
    
    c=sigmoid(X,W,w0)
    y_predicted=np.zeros(len(X))
    for a in range(len(X)):
        biggest=0
        biggest_value=c[a][0]
        
        for b in range(9):
            if c[a][b+1]>biggest_value:
                biggest=b+1
                biggest_value=c[a][b+1]

        y_predicted[a]=int(biggest+1)
        

        
        
            
            
    
    
    # your implementation ends above
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test)



# STEP 9
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, y_predicted):
    # your implementation starts below
    k=10
    h=len(y_truth)
    confusion_matrix=np.zeros((k,k))
    confusion_matrix=confusion_matrix.astype(int)
    for a in range(h):
        x=int(y_truth[a])
        y=int(y_predicted[a])
        
        

        confusion_matrix[x-1][y-1]+=1

            
 
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)
