#Melih Şahin ENGR 421 HW-2

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd

X_train = np.genfromtxt(fname="hw02_data_points.csv", delimiter=",", dtype=float)
y_train = np.genfromtxt(fname="hw02_class_labels.csv", delimiter=",", dtype=int)


# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_priors = np.zeros(5)
    number_of_things_in_each_class = np.zeros(5)
    h = 5000

    for a in range(h):
        for b in range(5):
            if y[a] == b + 1:
                number_of_things_in_each_class[b] += 1
    for c in range(5):
        class_priors[c] = number_of_things_in_each_class[c] / h

    # your implementation ends above
    return (class_priors)


class_priors = estimate_prior_probabilities(y_train)
print(class_priors)


# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    number_of_things_in_each_class = np.zeros(5)
    sample_means = np.zeros((5, 2))
    h = 5000

    for a in range(h):
        for b in range(5):
            if y[a] == b + 1:
                number_of_things_in_each_class[b] += 1
                sample_means[b][0] += X[a][0]
                sample_means[b][1] += X[a][1]

    for s in range(5):
        sample_means[s][0] /= number_of_things_in_each_class[s]
        sample_means[s][1] /= number_of_things_in_each_class[s]

    # your implementation ends above
    return (sample_means)


sample_means = estimate_class_means(X_train, y_train)
print(sample_means)


# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    sample_covariances=np.zeros((5,2,2))
    avarage_vectors=estimate_class_means(X,y)
    average_denominator=estimate_prior_probabilities(y)
    h=5000

    for a in range(h):
        for b in range(5):
            if y[a]==b+1:
                q=X[a]-avarage_vectors[b]
                
                matrix=np.zeros((2,2))
                matrix[0][0]=q[0]*q[0]
                matrix[0][1]=q[0]*q[1]
                matrix[1][0]=q[0]*q[1]
                matrix[1][1]=q[1]*q[1]
    

                sample_covariances[b]+=matrix*(1/(average_denominator[b]*h))

    # your implementation ends above
    return (sample_covariances)


sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)


# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    h=5000
    k=5
    score_values=np.zeros((h,k))
    
    for a in range(h):
        for b in range(k):
            first=(-1/2)*(np.log(np.linalg.det(class_covariances[b]))+2*(np.log(2*np.pi)))

            
            diffrence_vector=X[a]-class_means[b]
            inverse_matrix=np.linalg.inv(class_covariances[b])
            
            j=np.matmul(diffrence_vector, inverse_matrix)
            k1=j[0]*diffrence_vector[0]+j[1]*diffrence_vector[1]
            
            second=(-1/2)*k1

            

            third=np.log(class_priors[b])

            
            score_values[a][b]=first+second+third

    # your implementation ends above
    return (score_values)


scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)


# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    k=5
    h=5000
    confusion_matrix=np.zeros((k,k))
    for a in range(h):
        max1=0
        max_value=scores[a][0]
        for b in range(4):
            if scores[a][b+1]>max_value:
                max_value=scores[a][b+1]
                max1=b+1
        
        confusion_matrix[max1][y_truth[a]-1]+=1
        
            
            
    

    # your implementation ends above
    return (confusion_matrix)


confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)


def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:, :, c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis=2)

    fig = plt.figure(figsize=(6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize=2, markerfacecolor=class_colors[c], alpha=0.25,
                 markeredgecolor=class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize=4, markerfacecolor=class_colors[c],
                 markeredgecolor=class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return (fig)


fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches="tight")


# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below
    sample_covariances=np.zeros((5,2,2))
    h=5000
    new_y=np.copy(y)
    for a in range(h):
        new_y[a]=1
    q=estimate_class_covariances(X, new_y)
    
    for r in range(5):
        sample_covariances[r]=q[0]
        

        
    

    # your implementation ends above
    return (sample_covariances)


sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches="tight")