import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy import stats

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

# plt.scatter(pageSpeeds, purchaseAmount)
# plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
r_value ** 2


def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()