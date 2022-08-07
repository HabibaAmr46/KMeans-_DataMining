

import numpy as np
import pandas as pd
import random

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

class K_Means:
    def __init__(self, k=3, max_iterations=500): #initial
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data,IDs,type):

        self.centroids = {}

        init = random.sample(range(0,len(data)),self.k)
        # initialize the centroids, the random 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[init[i]]

        # begin iterations
        for i in range(self.max_iterations):
            self.classes = {} #Points
            self.ids={} #Ids of points
            for i in range(self.k): #key:Cluster number value:list
                self.classes[i] = []
                self.ids[i]= []

            # find the distance between the point and cluster; choose the nearest centroid
            for (features,m) in zip(data,IDs):
                if(type=='euc'):
                    distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                else:
                    distances = [manhattan(features, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances)) #min element in the distances array
                self.classes[classification].append(features)
                self.ids[classification].append(int(m))

            previous = dict(self.centroids)

            # average the cluster datapoints to re-calculate the centroids
            for classification in self.classes: #classification:clusters indices
                self.centroids[classification] = np.average(self.classes[classification],axis=0) #axis=0 along column
            isOptimal = True

            for centroid in self.centroids:#each centroid in the list according to k
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid))!=0:
                    isOptimal = False
            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            if isOptimal:
                break
        return self.ids,self.centroids,self.classes;


def main():
    data = pd.read_csv("Power_Consumption.csv")
    df1 = data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1','Sub_metering_2', 'Sub_metering_3']]
    df2=data[['Id']]#pandas frame
    X = df1.values  # returns a numpy array
    Y=  df2.values
    K=int(input("Enter the number of clusters"))
    Type = input("Enter the type of distance")
    km = K_Means(K)

    Clusters,Centroids,Classes = km.fit(X,Y,Type) #Ids,Centroids,Points
    for i in Clusters.keys():
        print("Cluster: ", i)
        print(Clusters[i])
    #----------------------------------
    points = []
    # distances will be used to calculate outliers
    distances = []

    for i in Centroids.keys():
        for c, u in zip(Classes[i], Clusters[i]): #Data and ids
            #Data and data ids
            if(Type=='euc'):
                distances = np.append(distances, np.linalg.norm(c - Centroids[i]))
            else:
                distances=np.append(distances,manhattan(c, Centroids[i]))
            points = np.append(points, u)

    PDistances=sorted(distances) #percentile Distances
    percentile = 90 # getting outliers whose distances are greater than some percentile
    outliers = points[np.where(distances > np.percentile(PDistances, percentile))]
    TotalOutliers=[]
    for i in outliers:
        TotalOutliers.append(int(i))
    print()
    print("Outliers",TotalOutliers)


main();
