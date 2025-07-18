import sympy as sp
import numpy as np

def w5_1():
    t = sp.symbols('t')
    x = sp.Function('x')
    v0 = 4
    m = 2
    alpha = np.radians(30)
    # x(t) в данному случае это dv/dt
    eqn = sp.Eq(m*sp.Derivative(x(t), t, 2), 12*sp.cos(3*t)-m*9.8*sp.sin(alpha))
    sp.pprint(eqn)
    # res = sp.dsolve(eqn, x(t), ics={x(t).subs(t, 0): v0})
    # sp.pprint(res)
    # vt = sp.integrate(res, t)
    # sp.pprint(vt)
    
def main():
    w5_1()

if __name__ == "__main__":
    main()
