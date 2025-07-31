import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

def w6_1():
    def F(x):
        if x <= 0:
            return 0
        elif 0 < x <= 0.5:
            return -4*x**2 + 4*x
        else:
            return 1

    print(F(1) - F(1/4))

def w6_3():
    paret = pareto(b=2.63)
    fig, ax = plt.subplots(1, 2)

    x=np.linspace(0, 10, 100)
    ax[0].plot(x, paret.pdf(x), lw=5, alpha=0.6)
    ax[1].plot(x, paret.cdf(x), lw=5, alpha=0.6)
    plt.show()

def main():
    w6_3()

if __name__ == "__main__":
    main()