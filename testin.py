import numpy as np

# Define the transformation matrices from option (a)

rotate = np.array([
        [np.sqrt(2)/2, -np.sqrt(2)/2, 0],
        [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
        [0, 0, 1]
    ])

scale = np.array([
        [np.sqrt(2), 0, 0],
        [0, np.sqrt(2), 0],
        [0, 0, 1]
    ])

translate= np.array([
        [1, 0, 3],
        [0, 1, 3],
        [0, 0, 1]
    ])

x = 0
y = 1

point = np.array([x, y, 1])

r = rotate @ point
s = translate @ r
t = scale @ s
print(t)