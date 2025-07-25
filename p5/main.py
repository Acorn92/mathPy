import sympy as sp
import numpy as np
from IPython.display import display, Math
from sympy.physics.vector import init_vprinting

init_vprinting(pretty_print=True)

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

def w5_5():
    m1, m2, R, g, alpha, t = sp.symbols('m1 m2 R g alpha t')
    x = sp.Function('x')(t)
    T = (0.5*(m1*(sp.diff(x, t)**2)) + 2*(0.5*(0.5*m1*R**2 *(sp.diff(x, t)/R)**2)) + 0.5*(m2*(sp.diff(x, t))**2)).simplify()
    sp.pprint(T)
    U = (-m1*g*x*sp.sin(alpha) + m2*g*x).simplify()
    sp.pprint(U)
    L = (T-U)
    sp.pprint(L)
    dLdx = sp.diff(L, x)
    sp.pprint(dLdx)
    dLdxx = sp.diff(L, sp.diff(x))
    sp.pprint(dLdxx)
    dLdxdt = sp.diff(dLdxx, t)
    sp.pprint(dLdxdt)

    eqL = sp.Eq(dLdxdt - dLdx, 0)
    sp.pprint(eqL)
    xxx = sp.solve(eqL, sp.diff(x, t, 2))
    sp.pprint(xxx[0])
    xx = sp.integrate(xxx[0].simplify(), t) + sp.symbols('v0')
    sp.pprint(xx)
    x = sp.integrate(xx, t) + sp.symbols('x0')
    sp.pprint(x)
def main():
    w5_5()

if __name__ == "__main__":
    main()
