import numpy as np

x = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
w = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])

y = np.dot(x,w)
print(y)


x1 = np.array([[1,2],[1,2],[1,2],[1,2]])
w1 = np.array([[1,2,3,4],[1,2,3,4]])

y1 = np.dot(x1,w1)
print(y1)

x2 = np.array([[3,4],[3,4],[3,4],[3,4]])
w2 = np.array([[1,2,3,4],[1,2,3,4]])

y2 = np.dot(x2,w2)
print(y2)

y3 = np.add(y1,y2)
print(y3)   