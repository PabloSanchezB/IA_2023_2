import numpy as np, random
"""""
coord = [0.0, 0.0]
for j in range(10):
    for i in range(2):
        coord[i] = random.uniform(-512, 512.0000001)
    print(coord)
"""

""""
x = 512
y = 404.2319
a = np.sqrt(np.fabs(y+x/2+47))
b = np.sqrt(np.fabs(x-(y+47)))
c = -(y+47)*np.sin(a)-x*np.sin(b)

print(c)
print(c * (-1))
"""

child = []

parent = [4, 5, 6, 8]

child = parent

print(child)


