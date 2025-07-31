import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.integrate import dblquad

def w6_1():
    def F(x):
        if x <= 0:
            return 0
        elif 0 < x <= 0.5:
            return -4*x**2 + 4*x
        else:
            return 1

    print(F(1) - F(1/4))

def w6_4():
    e = expon(scale = 4)
    print(e.var())

def w6_5():
  
    def func(x, y):
        C = np.array([0, 0])
        A = np.array([0, 4])
        B = np.array([4, 0])
        return (A + x * (B - A) + y * (C-A))[0]
    
    # print(func(0,4))
    print(dblquad(func, 0, 4, lambda x: 0, lambda x: 4 - x))

def main():
    w6_5()

if __name__ == "__main__":
    main()