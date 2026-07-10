import numpy as np
import matplotlib.pyplot as plt
import math
import time

start_time = time.perf_counter()

in_data = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], 
        [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
        [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

k = 5

def knn(k, in_data):

    # Test datapoint 
    test_datapoint = [4.3, 2.7]
    distance_list = []
    i = 0
    out_data_list = []
    #Calculate the distance between every input point and the test data point
    for points in in_data:
        x1 = points[0]
        y1 = points[1]
        x2 = test_datapoint[0]
        y2 = test_datapoint[1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_list.append([i,distance])
        i = i + 1

    #sort
    distance_array = np.array(distance_list)
    second_items = distance_array[:, 1]
    sorted_indices = np.argsort(second_items)
    sorted_arr = distance_array[sorted_indices]

    first_n_rows = sorted_arr[:k]
    print("\nK Nearest Neighbors:")
    #return the result
    i = 0
    for row in first_n_rows: 
        i += 1  
        print(i, "==>",in_data[int(row[0])])
        out_data_list.append(in_data[int(row[0])])
    out_data = np.array(out_data_list)
    return out_data
    
    # Visualize the nearest neighbors along with the test datapoint 
    

knn(k, in_data)

end_time = time.perf_counter()
run_time = end_time - start_time
print(f"time：{run_time:.4f} sec")