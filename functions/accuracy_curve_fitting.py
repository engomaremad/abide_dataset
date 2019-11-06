import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c
def func2(x, a, b, c):
    return a +np.multiply(b,x) + np.multiply(c,np.power(x,2))

def fit(x, y):
    plt.figure()
    plt.scatter(x, y, 'b-', label='data')
    popt, pcov = curve_fit(func, np.ravel(x), np.ravel(y))
    plt.plot(x, func(x, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    x = 0

