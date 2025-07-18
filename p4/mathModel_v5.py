
def w5_5(x):
    t = sp.symbols('t')
    O1A = O2B = 20
    R = 18

    q = (sp.pi*t**3)/6
    s = 6*sp.pi*t**2

    qM = s/R

    Ox_t = float((O1A*sp.cos(q)).subs(t,x).evalf())
    Oy_t = float((O1A*sp.sin(q)).subs(t,x).evalf()) 
    if (Oy_t < 0):
        Oy_t = np.abs(Oy_t)
    Ax_t = float((O1A*sp.cos(q)).subs(t,x).evalf()) + 2*R
    Ay_t = float((O1A*sp.sin(q)).subs(t,x).evalf())
    if (Ax_t < 0):
        Ax_t = np.abs(Ax_t)

    global collision
    if not collision:
        Mx_t = (Ox_t + R) + float((R*sp.cos(qM+ sp.pi*3/2)).subs(t, x).evalf())
        My_t = Oy_t  - float((R*sp.sin(qM + sp.pi*3/2)).subs(t, x).evalf())
    else:
        Mx_t = Ox_t
        My_t = Oy_t

    # VMx = 

    B = [Ox_t, Oy_t]
    M = [Mx_t, My_t]

    if (np.linalg.norm([M[0] - B[0], M[1] - B[1]]) < 0.1):
        collision = True


    rez = pd.Series(
    {
        't'   : t,  
        'R'   : 18,
        'O1O' : O1A,
        'O(t)' : [Ox_t, Oy_t],
        'A(t)' : [Ax_t, Ay_t],
        'M(t)' : [Mx_t, My_t]
        # 'VM'   :
    }
    )

    # print(rez['VeB'])
    # print(rez['VM'])
    return (rez)