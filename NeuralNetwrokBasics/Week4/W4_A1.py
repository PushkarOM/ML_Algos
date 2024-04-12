import numpy as np
import matplotlib.pyplot as plt
from public_tests import *


'''We need to craeted a binary decision tree to try and tell if a 
mushroom is edible or poisonous based on it's physical attributes'''


'''The examples vary as follows:-
    Three features:-
    cap color (brown or red)
    stalk shape (tapering or enlarging)
    solitary (yes or no)

    label:-
    Edible (1,0 for yes or no)
'''

'''For our ease we will turn the entire data into one hot encoded features.
So our traning data consist of X_train = 3 features
y_train = label
'''

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])


#printing the values in dataset
print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

#checking the deimension of the dataset
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

'''Decision Trees

steps of building a decision tree:-
1) start with all example at the root node
2) calculate information gain for splitting on all possible
features and pick th eon with highest informationm gain.
3)split dataset accordingly to the selected feautres, create left and right branches.
keep repeating the process till leaf node (final node)
'''



'''Calculating entorpy 
it is a measure of impurity at a node.
it takes an array y that indicates whether the examples in the
node are edible or not, compute the exmaples that are edible.

    h(p1) = -p1log2(p1) - (1-p1)log2(1-p1)

'''


def compute_entropy(y):
   
   
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """

    entorpy = 0.

    if len(y) != 0:
      
      p1 = len(y[y==1])/len(y)
      if p1 != 0 and p1 != 1:
         entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
      else:
         entropy = 0

    return entropy

# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"
print("Entropy at root node: ", compute_entropy(y_train))

'''Splitting the dataset
here we will create on the basis of wethere or not the mushroom
has brown cap.
'''


def split_dataset(X,node_indices,feature):
   """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list): Indices with feature value == 1
        right_indices (list): Indices with feature value == 0
    """
   left_indices = []
   right_indices = []

   for i in node_indices:
      if X[i][feature] == 1:
        left_indices.append(i)
      else:
        right_indices.append(i)      
    
   return left_indices,right_indices




root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("Left indices: ", left_indices)
print("Right indices: ", right_indices)




'''Calculating information gain

information gain = H(p1node) - (wleftH(p1left)) + (wrightH(p1right))

where:-
    h(p1node) -- entropy at node
    h(p1right or left) -- entropy at right or left branch
    wleft or wright -- are the proportion of examples on the left right branch
 '''


def compute_information_gain(X,y,node_indices,feature):
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """

    left_indices, right_indices = split_dataset(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0

    #node entropy & left and right
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    #proportion fo examples
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    
    #weighted entropy
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    
    #information gain
    information_gain = node_entropy - weighted_entropy

    return information_gain

#finding information gain on different features
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)
    
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

'''Getting the best split
Here we iterate through each feature to find the best
information gain
'''

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    num_features = X.shape[1]
    
    best_feature = -1

    max_info_gain=0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
   
       
    return best_feature


best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)



'''Bulding the Tree'''
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    tree.append((current_depth, branch_name, best_feature, node_indices))
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)


build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)

#above code's some part to be explained later
