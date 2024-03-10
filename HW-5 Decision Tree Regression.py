#Melih Şahin ENGR 421 HW5

import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

# get x and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 0.0
maximum_value = 2.0
step_size = 0.002
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below
    
    new_x=[]

    for r in range(len(X_train)):
        new_x.append(X_train[r][0])

    X_train=new_x

    # sort the X_train
    indices = np.argsort(X_train)
    X_train=np.sort(X_train)
    y_train=y_train[indices]





    is_terminal[1]=False
    node_indices[1] = (np.array(range(len(X_train))))
    need_split[1] = True


    def calculate_split_error(left, right, mean_left, mean_right):
        error=0
        for a in range(len(left)):
            error+=((y_train[left[a]]-mean_left)**2)

        for b in range(len(right)):
            error+=((y_train[right[b]]-mean_right)**2)

        return  error/(len(left)+len(right))

    def do_split(node_index):
        split_errors = [None]*(len(node_indices[node_index]))
        for a in range(len(node_indices[node_index])): # check for all possible splits
            current_slit_left_data_indices=[]
            current_slit_right_data_indices=[]

            for b in range(a):
                current_slit_left_data_indices.append((node_indices[node_index])[b])
            for c in range(len(node_indices[node_index])-a):
                current_slit_right_data_indices.append((node_indices[node_index])[a+c])


            mean_of_left=calculate_node_mean(current_slit_left_data_indices)

            mean_of_right=calculate_node_mean(current_slit_right_data_indices)


            the_split_error=calculate_split_error(current_slit_left_data_indices,current_slit_right_data_indices,mean_of_left,mean_of_right )

            split_errors[a]=the_split_error;
            
        min_index=np.argmin(split_errors)
        left_node=[]
        right_node=[]
        
        


        for u in range(min_index):
            left_node.append((node_indices[node_index])[u]) # we assume the data is ordered, if the first root node is ordered the remains no problem
        for u1 in range(len(node_indices[node_index])-min_index):
            right_node.append((node_indices[node_index])[u1+min_index])
            
        #print("****")
        #print(len(node_indices[node_index]))
        #print(min_index)
        #print(calculate_node_mean(left_node))
        #print(calculate_node_mean(right_node))
        #print(X_train[node_indices[node_index][min_index]])
        #print("****1")
        

        return left_node, right_node, (X_train[node_indices[node_index][min_index]]+X_train[node_indices[node_index][min_index]-1])/2,min_index

    def calculate_node_mean(node):

        if (len(node)==0):
            return 0
        mean=0
        for a in range(len(node)):
            mean+=y_train[node[a]]
        mean=mean/len(node)
        return mean

    def recursive_program(node_index):
        

        left, right, split_position, min_index=do_split(node_index)
        if (min_index==0):
            is_terminal[node_index]=True
            node_features[node_index] = 0
        else:
            node_features[node_index] = 0
            node_splits[node_index]=split_position

            node_indices[2*node_index]=left
            node_indices[2*node_index+1] = right
  
            node_means[2*node_index]=calculate_node_mean(left)
            node_means[2 * node_index+1]=calculate_node_mean(right)

            size_of_left=len(left)

            size_of_right=len(right)

            if (size_of_left)<=P:
                is_terminal[2*node_index]=True
            else:
                is_terminal[2 * node_index] = False
                recursive_program(2*node_index)


            if (size_of_right)<=P:
                is_terminal[2*node_index+1]=True
            else:
                is_terminal[2 * node_index+1] = False
                recursive_program(2*node_index+1)

    node_means[1]=calculate_node_mean(node_indices[1])
    recursive_program(1)
    
    
    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)

# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    y_hat=[]
    for a in range(len(X_query)):
        current_node_index=1
        while (not (is_terminal[current_node_index])):
            #if X_query[a][0]==1.402778:
                #print("---------------------------------")
                #print(current_node_index)
                #print(node_splits[current_node_index])
            a1=node_splits[current_node_index]
            if X_query[a][0]<=a1:
                current_node_index=2*current_node_index
            else:
                current_node_index = 2 * current_node_index +1
            
            

        #if X_query[a][0]==1.402778:
            #print(node_means[current_node_index])
        y_hat.append(node_means[current_node_index])


    
    # your implementation ends above
    return(y_hat)

# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    string5=[]
    string_indexes=[]
    
    
    def recursive(node_key, node_array, string, recursion_depth):
  
        if is_terminal[node_key]:

            string3=" ["
            for j in range(len(node_array)):
                string3+="’x1 "+string[j]+" "+str(node_array[j])+"’"
            string3+="]"
            a="Node "+str(node_key)
            string3=a+string3+" => "+str(node_means[node_key])
            string5.append(string3)
            string_indexes.append(recursion_depth)
                
        else:

            node_array1=node_array.copy();
            node_array1.append(node_splits[node_key])
            
            string1=string.copy()
            string2=string.copy()
            
            string1.append("<=")
            string2.append(">")
            
            recursive((2*node_key),node_array1,string1,recursion_depth+1)
            recursive((2*node_key)+1,node_array1,string2, recursion_depth+1)
            
    recursive(1,[],[],0)
    
    indices = np.argsort(string_indexes)  
        
    for t in range(len(string5)):
        k=indices[t]
        print(string5[k])

    
    # your implementation ends above

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))


extract_rule_sets(is_terminal, node_features, node_splits, node_means)
