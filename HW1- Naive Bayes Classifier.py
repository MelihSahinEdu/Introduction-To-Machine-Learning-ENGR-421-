#Melih Şahin ENGR 421 HW-1

import numpy as np
import pandas as pd
X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)
# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train=X[0:50000]
    y_train=y[0:50000]
    X_test=X[50000: 93925]
    y_test=y[50000: 93925]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)
X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_priors=np.zeros(2)
    number=0
h=y.size
    for a in range(h):
        if y[a]==1:
            number+=1
    class_priors[0]=number/h
    class_priors[1]=1-class_priors[0]
    # your implementation ends above
    return(class_priors)
class_priors = estimate_prior_probabilities(y_train)
print(class_priors)
# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
h=y.size
    A1=np.zeros(7)
    A2=np.zeros(7)
    C1=np.zeros(7)
    C2=np.zeros(7)
    G1=np.zeros(7)
    G2=np.zeros(7)
    T1=np.zeros(7)
    T2=np.zeros(7)
    class1=0
    class2=0
     for c in range(h):
        if y[c]==1:
            class1+=1
        else:
            class2+=1
    for a in range(h):
        for b in range(7):
            if X[a][b]=="A":
                if y[a]==1:
                    A1[b]+=1
                else:
A2[b]+=1
            elif X[a][b]=="C":
                if y[a]==1:
                    C1[b]+=1
                else:
C2[b]+=1
            elif X[a][b]=="G":
                if y[a]==1:
                    G1[b]+=1
                else:
G2[b]+=1
            elif X[a][b]=="T":
                if y[a]==1:
T1[b]+=1
                else:
                    T2[b]+=1
    A1 = [i * 1/class1 for i in A1]
    T1 = [i * 1/class1 for i in T1]
    C1 = [i * 1/class1 for i in C1]
    G1 = [i * 1/class1 for i in G1]
    A2 = [i * 1/class2 for i in A2]
    T2 = [i * 1/class2 for i in T2]
    C2 = [i * 1/class2 for i in C2]
    G2 = [i * 1/class2 for i in G2]
    pAcd=[A1,A2]
    pCcd=[C1,C2]
    pGcd=[G1,G2]
    pTcd=[T1,T2]
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)
pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)
# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    h=int((X.size)/7)
    score_values=np.zeros((h,2))
    for a in range(h):
        number=1
        c1=np.zeros(2)
        for b in range(7): #category 1
            c=X[a][b]

             if c=="A":
                number*=pAcd[0][b]
            elif c=="C":
                number*=pCcd[0][b]
            elif c=="G":
                number*=pGcd[0][b]
            elif c=="T":
                number*=pTcd[0][b]
        number=np.log(number)
        number+=np.log(class_priors[0])
        c1[0]=number
        number=1
        for b in range(7): #category 2
            c=X[a][b]
            if c=="A":
                number*=pAcd[1][b]
            elif c=="C":
                number*=pCcd[1][b]
            elif c=="G":
                number*=pGcd[1][b]
            elif c=="T":
                number*=pTcd[1][b]
        number=np.log(number)
        number+=np.log(class_priors[1])
        c1[1]=number
        score_values[a]=c1
    # your implementation ends above
    return(score_values)
scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)
scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)
# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    confusion_matrix=np.zeros((2,2))
    h=y_truth.size
    for a in range(h):
        a1=0
        if (scores[a][0]>scores[a][1]):
a1=1 else:
            a1=2
        b=y_truth[a]
        if a1==1:
            if b==1:
                confusion_matrix[0][0]+=1
            else:
                confusion_matrix[0][1]+=1
        else:
            if b==1:
                confusion_matrix[1][0]+=1
            else:
                confusion_matrix[1][1]+=1
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)