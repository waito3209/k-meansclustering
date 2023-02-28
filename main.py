import matplotlib.pyplot as plt
import numpy as np
import math
from KMC import *
from KmaxC import *

from KmedC import *
data = []
r=50
for i in range(-r,r):
    for j in range(-r,r):
        if math.sqrt(i*i +j*j)<r:
            data.append(np.asarray([i,j]))
data = np.asarray(data)

# for i in range(10):
#     try:
#         kmed(data,7,100,20,-r,r,'kmedclust',0.5)
#     except:
#         pass
for i in range(10):
    try:
        kmax(data,7,50,20,-r,r,'data',0.5)
    except:
        pass