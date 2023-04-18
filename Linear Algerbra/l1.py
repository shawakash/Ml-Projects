import numpy as np

v = np.array([[1,2,3],[4,5,6],[7,8,10]])
w = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(np.add(v,w))
print(np.subtract(v,w))
print(v*69)
print(np.linalg.det(v))
print(np.dot(v,v))
print(np.linalg.inv(np.dot(v,v)))
print(np.linalg.inv(w))
print(np.linalg.det(np.linalg.inv(np.dot((input("Give Ist Matrix")), (input("Give 2nd Matrix"))))))