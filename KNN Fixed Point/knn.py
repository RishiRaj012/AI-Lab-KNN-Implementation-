import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def generate_k(n_samples):
    k = int(np.sqrt(n_samples))
    if k % 2 == 0:
        k += 1
    return max(1, k)


def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]


training_data = [[22, 24000], [26, 25500], [21, 23000], [27, 26000], [24, 24500],
                 [23, 25000], [25, 23500], [28, 27000], [20, 22000], [29, 28000]]
training_labels = ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C']
test_point = [24, 25000]

k = generate_k(len(training_data))
prediction = knn_predict(training_data, training_labels, test_point, k)
print(f"Predicted Label for test point {test_point} is: {prediction}")


label_colors = {'A': 'red', 'B': 'green', 'C': 'blue'}


for i in range(len(training_data)):
    plt.scatter(training_data[i][0], training_data[i][1],
                color=label_colors[training_labels[i]], label=training_labels[i] if i == training_labels.index(training_labels[i]) else "")


plt.scatter(test_point[0], test_point[1], color='black', label='Test Point', marker='X', s=100)

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('KNN Classification')
plt.legend()
plt.grid(True)
plt.show()
