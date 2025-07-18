import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

collision = False

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

def v4_5(x):
    # print(x)
    t = sp.symbols('t')
    O1A = O2B = 20
    R = 18
    # s = sp.pi*(t**2)
    # q = 5/48 * sp.pi * t**3
    
    q = (sp.pi*t**3)/6
    s = 6*sp.pi*t**2

    w = sp.diff(q)
    e = sp.diff(w)
    #находим углы
    qO1_2 = float(q.subs(t,2).evalf())
    s_2 = float(s.subs(t,2).evalf())
    qM_2 = s_2/R
    qm = (s/R)
    #находим скорости
    VeM = float((O1A*w).subs(t, x).evalf()) #переносная скорость М
    VeAx = float((O1A*sp.cos(q)).subs(t,x).evalf())
    VeAy = float((O1A*sp.sin(q)).subs(t,x).evalf()) 
    if (VeAy < 0):
        VeAy = np.abs(VeAy)
    VeBx = float((O1A*sp.cos(q)).subs(t,x).evalf()) + 2*R
    VeBy = float((O1A*sp.sin(q)).subs(t,x).evalf())
    if (VeBy < 0):
        VeBy = np.abs(VeBy)
    # VeA = 
    VrM = float(sp.diff(s).subs(t, x).evalf())
    #разница углов даёт угол между векторами
    VvMx = np.sqrt(VeAx**2 + VrM**2 + 2*VrM*VeAx*np.cos(qO1_2 - qM_2))
    # VMy = np.sqrt(VeAy**2 + VrM**2 + 2*VrM*VeAy*np.cos(qO1_2 - qM_2))
    # VMy = float(s.subs(t, x).evalf())
    global collision
    if not collision:
        VMx = (VeAx + R) + float((R*sp.cos(qm + sp.pi*3/2)).subs(t, x).evalf())
        VMy = VeAy  - float((R*sp.sin(qm + sp.pi*3/2)).subs(t, x).evalf())
    else:
        VMx = VeBx
        VMy = VeBy
    B = [VeBx, VeBy]
    M = [VMx, VMy]

    VvMx = float(sp.diff(VeAx + (15*sp.cos(qm - sp.pi/2))).subs(t, x).evalf())
    VvMy = float(sp.diff((VeAx + 15) + (15*sp.sin(sp.pi*3/2 + qm))).subs(t, x).evalf())
   
    
    if (np.linalg.norm([M[0] - B[0], M[1] - B[1]]) < 0.1):
        collision = True
    # print(sp.sin((q + qm + sp.pi/2) - 2*sp.pi).subs(t, x).evalf())

    # sp.pprint(VM)

    AetM = float((e*O1A).subs(t,2).evalf())
    AenM = float(((w**2) * O1A).subs(t, 2).evalf())

    ArtM = float(sp.diff(sp.diff(s)).subs(t,2).evalf())
    ArnM = (VrM**2) / 15
    AcM = float((2*(w*VrM*np.sin(np.pi/2 - qO1_2 + qM_2))).subs(t,2).evalf())

    AM = np.sqrt((AetM + ArtM)**2 + (AenM + ArnM + AcM)**2)
    # sp.pprint(AetM)
    # sp.pprint(AenM)
    # sp.pprint(ArtM)
    # sp.pprint(ArnM)
    # sp.pprint(AcM)
    # sp.pprint(AM)
    

    rez = pd.Series(
        {
            't'   : t,  
            'R'   : 18,
            'O1A' : O1A,
            'VeA' : [VeAx, VeAy],
            'VeB' : [VeBx, VeBy],
            'VM'  : [VMx, VMy],  
            'VvM' : [VvMx, VvMy]
        }
    )

    # print(rez['VeB'])
    # print(rez['VM'])
    return (rez)


def w5_5(x):
    t = sp.symbols('t')
    O1A = O2B = 20
    R = 18

    

    q = (sp.pi*t**3)/6
    s = 6*sp.pi*t**2

    tA = float(sp.solve(sp.Eq(s/R, sp.pi), t)[1].evalf())+0.1
    # print(s.subs(t,tA))
    qM = s/R

    O = [O1A*sp.cos(q), O1A*sp.sin(q)]
    A = [O1A*sp.cos(q) + 2*R, O1A*sp.sin(q)]

    if not collision:
        M = [(O[0] + R) + R*sp.sin(qM), O[1] + R*sp.cos(qM)]
    else:
        M = A

    w = sp.diff(s)
    sp.pprint((w).subs(t,1))
    # wO = [sp.diff(O[0]), sp.diff(O[1])]
    wO = [sp.diff(q)*s*sp.sin(q), sp.diff(q)*s*sp.cos(q)]
    VrM = [sp.diff(M[0]) + O1A, sp.diff(M[1]) + O1A]
    VaM = [wO[0] + VrM[0], wO[1] + VrM[1]]
    sp.pprint(VrM)
    sp.pprint(wO)

    def count(x):


        rez = pd.Series(
        {
            't'   : t,  
            'R'   : 18,
            'O1O' : O1A,
            'tA'  : tA, 
            'O(t)' : np.array([float(O[0].subs(t,x).evalf()), float(O[1].subs(t,x).evalf()), 0]),
            'A(t)' : [float(A[0].subs(t,x).evalf()), float(A[1].subs(t,x).evalf())],
            'M(t)' : [float(M[0].subs(t,x).evalf()), float(M[1].subs(t,x).evalf())],
            # 'VeO'  : np.array([float(wO[0].subs(t,x).evalf()), float(wO[1].subs(t,x).evalf()), 0])*0.0025,
            'VrM'  : np.array([float(VrM[0].subs(t,x).evalf()), float(VrM[1].subs(t,x).evalf()), 0])*0.0025,
            'VaM'  : np.array([float(VaM[0].subs(t,x).evalf()), float(VaM[1].subs(t,x).evalf()), 0])*0.0025,
        }
        )

        global collision
        if (np.linalg.norm([rez['M(t)'][0] - rez['A(t)'][0], rez['M(t)'][1] - rez['A(t)'][1]]) < 1):
            collision = True

        if (rez['O(t)'][1] < 0):
            rez['O(t)'][1] = np.abs(rez['O(t)'][1])
        if (rez['A(t)'][1] < 0):
            rez['A(t)'][1] = np.abs(rez['A(t)'][1])

        print(rez['O(t)'])
        return (rez)
    return count(x)

import math
def rotate(angle, o):
    rez = sp.Matrix()
    if o == 'x':
        rez = sp.Matrix([[1, 0, 0, 0],
                         [0, sp.cos(angle), -sp.sin(angle), 0],
                         [0, sp.sin(angle), sp.cos(angle),0],
                         [0, 0, 0, 1]])
    elif o == 'y':
        rez = sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), 0],
                         [0, 1, 0, 0],
                         [-sp.sin(angle), 0, sp.cos(angle), 0],
                         [0, 0, 0, 1]])
    elif o == 'z':
        rez = sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0, 0],
                         [sp.sin(angle), sp.cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return rez

def shift(shift, o):
    rez = sp.Matrix()
    if o == 'x':
        rez = sp.Matrix([[1, 0, 0, shift],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif o == 'y':
        rez = sp.Matrix([[1, 0, 0, 0],[0, 1, 0, shift], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif o == 'z':
        rez = sp.Matrix([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, shift], [0, 0, 0, 1]])
    return rez

def w4_6():
    d1 = sp.Symbol('d1')
    d2 = sp.Symbol('d2')
    d3 = sp.Symbol('d3')
    s = sp.Symbol('s')
    an1 = sp.Symbol('theta1')
    an2 = sp.Symbol('theta2')
    an3 = sp.Symbol('theta3')
    
    # H = rotate(an1, 'z')*rotate(-sp.pi/2, 'y')*shift(d1, 'x')*rotate(sp.pi/2, 'y')*shift(d2,'x')*rotate(sp.pi/2, 'z')*shift(d3, 'x')*rotate(sp.pi, 'y')
    # H = rotate(an1,'z')*rotate(-sp.pi/2, 'y')*shift(d1,'x')*rotate(an2,'z')*rotate(sp.pi/2,'z')*shift(r, 'x')*shift(d3, 'x')*rotate(sp.pi/2,'x')*rotate(-sp.pi/2,'z')
    H = rotate(an1,'z')*shift(d1, 'z')*rotate(an2, 'y')*shift(d3, 'x')*shift(s, 'z')
    sp.pprint(H[7])
    Q = sp.Quaternion.from_rotation_matrix(H[0:3,0:3])
    an = Q.to_euler('zyx')
    sp.pprint(an)


def w4_7():
    d1 = sp.Symbol('d1')
    d2 = sp.Symbol('d2')
    d3 = sp.Symbol('d3')
    r = sp.Symbol('r')
    an1 = sp.Symbol('theta1')
    an2 = sp.Symbol('theta2')
    an3 = sp.Symbol('theta3')

    H = rotate(an1, 'z')*shift(d1,'z')*shift(d2, 'x')*rotate(an2, 'z')*shift(d3, 'x')*rotate(sp.pi*3/2,'x')
    sp.pprint(H)
    Q = sp.Quaternion.from_rotation_matrix(H[0:3,0:3])
    an = Q.to_euler('zyx')
    # sp.pprint(an)
def main():
    w4_6()

if __name__ == "__main__":
    main()
