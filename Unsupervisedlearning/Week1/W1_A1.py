import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''K-means Algorithm

It is an method that automatically cluster similer
together.
For example you are give a dataset {x1,x2,x3....xm}
and you want to group the data in a few cohesive cluster.
K-means helps you do that.It is an iterative procedure.
First,we give it totally random values of initial centroids or 
simply co=ord of center of cricle, then
Second,We check which datapoints are closer to which cluster,then
Third,we re-calculate or update the centroides on the basis of the above.

In short this is how K-means work.
'''




#creating the function to find the closest centroids

def find_closest_centroids(X,centroids):
    """
    Computes the 'closest centorids for every example'

    Args:
        X (ndarry): (m,n) Input values
        centorids (ndarry) : K centroids  k -- number of cluster you want to create

    Returns:
        idx (array_like): (m,) closest centorids
    
    """

    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype = int)

    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i]-centroids[j])    #mathematically to denote the distance between 2 points we write  
                                                           #it as ||x^i - uk||, it is called as L2-Norm.
            distance.append(norm_ij)  #finding the smallest L2-Norm or find the smallest distance between the given point and centorides

        idx[i] = np.argmin(distance)   #this picks out the indice of the smallest distance value

    return idx


X = load_data()
print("First five element of X are: ",X[:5])
print('The shape of X is: ',X.shape)


initial_centorids = np.array([[3,3],[6,2],[8,5]])
idx = find_closest_centroids(X, initial_centorids)

print("First three elements in ids are: ",idx[:3])




def compute_centroids(X, idx, K):
    """
    Returns the new Assigned centroids by computing the mean
    of the data points assigned to each centroid

    Args:
        X (ndarray): (m, n) Data Points
        idx (ndarray): (m,) Array contaning index of closet centorid for each
                        example in X.more prcisely, idx[i] contains the index of
                        the centorid closet to example i
        k (int) : number of cnetorids
    
    Returns:
        centroids (ndarray): (K, n) New Centroids computed
    
    """


    m, n = X.shape

    centroids = np.zeros((K,n))

    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points,axis=0)

    return centroids





K = 3
centroids = compute_centroids(X, idx, K)
print("The Centroids are: ", centroids)




'''Now that we have implemented these basic function of the
k-mean algorithm, let's see how this all works in a loop to
continuilly and iterativly change the value of the centorids 
and find the best one'''


def run_kMeans(X, initial_centorids, max_iter = 10, plot_progress=False):
    """
    Runs K-means algorithm on data matrix X, where row of X 
    is a single example.
    """


    m,n = X.shape
    K = initial_centorids.shape[0]
    centroids = initial_centorids
    previous_centorids = centroids
    idx = np.zeros(m)


    for i in range(max_iter):
        idx = find_closest_centroids(X, centroids)

        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iter-1))

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx




# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = 3

# Number of iterations
max_iters = 10

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)


'''Now that we have seen the k-Means run lets,now create a funcition
to randomly initialize the value of initial_centroids.
It is a good practise to set these value to random example values'''


def kMeans_init_centroids(X,K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """  

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids