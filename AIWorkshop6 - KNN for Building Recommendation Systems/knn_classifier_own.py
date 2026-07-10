import numpy as np
import matplotlib.pyplot as plt
import math

def knn(k, in_data,test_datapoint):
    in_data, y = data[:, :-1], data[:, -1].astype(int)
    # Test datapoint 
    distance_list = []
    i = 0
    #Calculate the distance between every input point and the test data point

    for i, points in enumerate(in_data):
        x1, y1 = points[0], points[1]
        x2, y2 = test_datapoint[0], test_datapoint[1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)** 2)
        distance_list.append([i, distance]) 

    #sort
    distance_array = np.array(distance_list)
    second_items = distance_array[:, 1]
    sorted_indices = np.argsort(second_items)
    sorted_arr = distance_array[sorted_indices]
    first_n_rows = sorted_arr[:k]
    original_indices = [int(row[0]) for row in first_n_rows]
    corresponding_labels = y[original_indices] 
    label_counts = np.bincount(corresponding_labels)
    
    #return the result
    max_count = np.max(label_counts)  
    max_count_labels = np.where(label_counts == max_count)[0] 
    print("Predicted output:",max_count_labels[0])


# Load input data
input_file = 'data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)


#set a test datapoint
test_datapoint = [5.1, 3.6]
knn(12,data,test_datapoint)