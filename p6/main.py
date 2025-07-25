import numpy as np
import matplotlib.pyplot as plt


def w6_1():
    def F(x):
        if x <= 0:
            return 0
        elif 0 < x <= 0.5:
            return -4*x**2 + 4*x
        else:
            return 1

    print(F(1) - F(1/4))

def main():
    w6_1()

if __name__ == "__main__":
    main()