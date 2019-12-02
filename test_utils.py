from utils import bb_intersection_over_union, iou_mat
import numpy as np

listBoxA = []

for i in range(5):
    x_min = np.random.randint(50 - 1)
    y_min = np.random.randint(50 - 1)
    x_max = np.random.randint(x_min + 1, 50)
    y_max = np.random.randint(y_min + 1, 50)
    listBoxA.append(np.array([x_min, y_min, x_max, y_max]))

listBoxB = []

for i in range(6):
    x_min = np.random.randint(50 - 1)
    y_min = np.random.randint(50 - 1)
    x_max = np.random.randint(x_min + 1, 50)
    y_max = np.random.randint(y_min + 1, 50)
    listBoxB.append(np.array([x_min, y_min, x_max, y_max]))

iou_matrix = iou_mat(listBoxA, listBoxB)

print(iou_matrix)

print(bb_intersection_over_union(listBoxA[0], listBoxB[0]))
