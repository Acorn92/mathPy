import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
#
# к Видео 4.2
# задача 1
#
def v4_2():
    t = sp.symbols('t')
    O = sp.Matrix([5*t-1, 4*t**3 +1])

    sp.pprint(O)

    V = sp.diff(O, t)
    a = sp.diff(V, t)

    sp.pprint(V)
    sp.pprint(a)

    at = a.dot(V)/V.norm() * V/V.norm()
    sp.pprint(at)

    an = (a - at).norm()
    sp.pprint(an)

    an_v = []

    for i in range(0, 6):
        an_v.append(an.subs(t,i).evalf())

    print(max(an_v))

#
# Видео 4.3
# Задача 3
#
def v4_3():
    t, r2, R2, r3, R3 = sp.symbols('t r2 R2 r3 R3')
    # t = np.linspace(-10, 10, 100)
    xt = 2 + 70*t**2
    x0 = xt.subs(t, 0)
    sp.pprint(x0)
    w1 = sp.diff(xt)
    sp.pprint(w1)
    wr2 = w1/r2
    sp.pprint(wr2)
    vR2 = wr2*R2
    sp.pprint(vR2)
    wR3 = vR2/R3
    sp.pprint(wR3)
    vM = wR3*r3
    sp.pprint(vM)

def v4_4():
    params = {
        'aA' : 2,
        'aB' : 4*np.sqrt(2),
        'd' : 2
    }
    aBx = params['aB']*np.cos(np.pi/4)
    aBy = params['aB']*np.sin(np.pi/4)

    aA = sp.Matrix([0, params['aA']])
    aB = sp.Matrix([aBx, aBy])
    rABc = sp.Matrix([2, 0, 0])
    rBCc = sp.Matrix([0, 2, 0])
    rAB = sp.Matrix([2, 0])
    e, w = sp.symbols('e w')
    eM = sp.Matrix([0,0,e]) #угловое ускорение
    eq = aA + eM.cross(rABc)[0:2, 0] + (w**2)*rAB - aB #формула распределения ускорений в твердом теле
    #через вектор к которому приложено ускорение левой части
    w2 = 2
    E = sp.solve(eq, e)
    eqC = aA + eM.subs('e', 1).cross(rBCc)[0:2, 0] + w2*rAB #для ускорения аС через вектор ВС
    
    sp.pprint(eqC.norm(1)) 

def main():
    v4_4()

if __name__ == "__main__":
    main()
