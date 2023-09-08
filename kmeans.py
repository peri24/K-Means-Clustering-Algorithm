import numpy as np
import matplotlib.pyplot as plt
import random
#Initialize k value
k = 2


#Read in data points from kmeans.cvs
data = np.genfromtxt('kmeans.csv', dtype=int, skip_header=1 , delimiter=',')



#Pick points as initial centroids at random
def pick_centroids(data, k):
    first_centroids = random.sample(list(data), k)
    return first_centroids



#Plot inital points and randomly chosen centroids
def plot_initial(cluster, centroids):
    x = []
    y = []
    for i in range(len(cluster)):
        x.append(cluster[i][0])
        y.append(cluster[i][1])

    cX = []
    cY = []
    for i in range(len(centroids)):
        cX.append(centroids[i][0])
        cY.append(centroids[i][1])
    
    plt.scatter(x, y, color= "red", s=30)
    plt.scatter(cX, cY,  color= "black", marker= "*", s= 100 )

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('K means initial')
    plt.show()

    return



#Plot final clusters and final centroids
def plot_final(cluster1, cluster2, centroids):

    #Plot final cluster 1
    clust1x = []
    clust1y = []
    for i in range(len(cluster1)):
       clust1x.append(cluster1[i][0])
       clust1y.append(cluster1[i][1])
    
    #Plot final cluster 2
    clust2x = []
    clust2y = []
    for i in range(len(cluster2)):
        clust2x.append(cluster2[i][0])
        clust2y.append(cluster2[i][1])

    #Plot final centroids
    centX = []
    centY = []
    for i in range(len(centroids)):
        centX.append(centroids[i][0])
        centY.append(centroids[i][1])
    
    plt.scatter(clust1x, clust1y, label = "Cluster1",  color= "red", s=30)
    plt.scatter(clust2x, clust2y, label = "Cluster2",  color= "blue", s=30)
    plt.scatter(centX, centY,  color= "black", marker= "*", s= 50 )

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('K means final')
    plt.legend()
    plt.show()

    return



#Label data points for clustering by indicating a "0" for cluster1 and a "1" for cluster2
def assign_clusters(data, centroids):

    #Cluster status array
    cluster_status = np.zeros(len(data))
    #Euclidean distances from centroids array size 2x30
    distances = [[0]* len(data) for x in range(len(centroids))]

    i = 0  
    j = 0
    for centroid in centroids:
        j = 0
        for point in data:
            distances[i][j] = round(np.sqrt(np.sum((centroid[0] - point[0]) ** 2) + ((centroid[1] - point[1]) ** 2)), 4)
            j += 1
        i += 1

    #Compare distance of each data point from both centroids, if distance from centroid 1 is greater than distiance from cetnroid 2 assign point to centroid 2
    for x in range(len(data)):
        if distances[0][x] > distances[1][x]:
            cluster_status[x] = 1

    #Return status array
    return cluster_status



#Update Centroids based on current clusters
def update_centroids(data, cluster_status, centroids, k):
    
    #Array for new centroids
    new_centroids = [[0]* k for x in range(len(centroids))]
    cluster1 = []
    cluster2 = []

    #Sort points into seperate cluster arrays
    for i in range(len(cluster_status)):
        if cluster_status[i] == 0:
            cluster1.append(data[i])
        elif cluster_status[i] == 1:
            cluster2.append(data[i])

    #Calculate new centroid for cluster 1
    for i in range(len(cluster1)):
        new_centroids[0][0] += cluster1[i][0] 
        new_centroids[0][1] += cluster1[i][1]
    
    new_centroids[0][0] = new_centroids[0][0] / len(cluster1)
    new_centroids[0][1] = new_centroids[0][1] / len(cluster1)

    #Calculate new centroids for cluster 2
    for i in range(len(cluster2)):
        new_centroids[1][0] += cluster2[i][0] 
        new_centroids[1][1] += cluster2[i][1]
    
    new_centroids[1][0] = round(new_centroids[1][0] / len(cluster2),2)
    new_centroids[1][1] = round(new_centroids[1][1] / len(cluster2),2)

    return new_centroids , cluster1, cluster2



#Kmeans clustering algorithm
def kmeans(data, k):

    #Pick initial centroids
    centroids = pick_centroids(data, k)
    #Plot initial points
    plot_initial(data, centroids)

    #Create new and old centroids tracker
    new_centroids = centroids.copy()
    temp_centroids = centroids.copy()

    print("\n")
    print("FIRST CENTROIDS:", centroids)

    max_interations = 20
    iter_count = 0

    #Iterate clustering algorithm until New centroids is equal to old centroids
    for i in range(max_interations):
        iter_count += 1
        print("\n")
        print("-".center(100, "-"))
        print("Iteration Count:", iter_count)
        print("\n")

        cluster_status = assign_clusters(data, centroids)
        temp_centroids = new_centroids.copy()
        new_centroids, cluster1, cluster2 = update_centroids(data, cluster_status, centroids, k)

        print("NEW CENTROIDS")
        print(np.around(new_centroids, decimals=2))

        print("\n")
        print("Cluster 1 length:", len(cluster1))
        print("Cluster 2 length:", len(cluster2))
        print("\n")
        print("Cluster 1: ",cluster1)
        print("\n")
        print("Cluster 2: ",cluster2)                                           
        
        centroids = new_centroids.copy()

        #When new and old centroids match, print final clusters and final centroids
        if np.array_equal(new_centroids,temp_centroids):
            print("\n")
            print("FINAL CENTROIDS:")
            print(np.around(centroids, decimals=2))
            plot_final(cluster1,cluster2,centroids)
            break


kmeans(data, k)