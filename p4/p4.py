import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import matplotlib.patches as pltp
from matplotlib.animation import FuncAnimation, PillowWriter

params = {
    'wO1A'  : 2,
    'q'     : np.radians(60),
    'a'     : 56,
    'b'     : 10,
    'c'     : 26,
    'd'     : 16,
    'e'     : 25,
    'O1A'   : 21,
    'O2B'   : 25,
    'O3F'   : 20,
    'AB'    : 54,
    'BC'    : 52,
    'CD'    : 69,
    'CE'    : 35,
    'EF'    : 32
}

#неподвижные точки
O1 = np.array([0,0])
O2 = np.array([params['a'], -params['c']])
O3 = np.array([params['a'] + params['b'], params['d'] + params['e']])

t = sp.symbols('t', real=True)

def cl_intersection(O, r, A, B, C):
    C1 = C + A * O[0] + B * O[1]
    norm = (A**2 + B**2)
    d = r**2 - C1**2/norm
    mult = sp.sqrt(d/norm)

    x0 = - A * C1 / norm
    y0 = - B * C1 / norm

    P1 = sp.Point2D([x0 + B * mult + O[0], y0 - A * mult + O[1]])
    P2 = sp.Point2D([x0 - B * mult + O[0], y0 + A*mult + O[1]])

    return (P1, P2)

def c_intersection(O1, r1, O2, r2):
    A = -2 * (O2[0] - O1[0])
    B = -2 * (O2[1] - O1[1])
    C = (O2[0] - O1[0])**2 + (O2[1] - O1[1])**2 + r1**2 - r2**2
    (P1, P2) = cl_intersection(sp.Point2D(0, 0), r1, A, B, C)
    (P1, P2) = (sp.Point2D([P1[0] + O1[0], P1[1] + O1[1]]), sp.Point2D([P2[0] + O1[0], P2[1] + O1[1]]))
    return (P1, P2)

def interS_Point(point1, point2, vector_list):
    x, y = sp.symbols('x y', real=True)
    eq1 = sp.Eq((x - point1[0])**2 + (y - point1[1])**2, vector_list[0]**2)
    eq2 = sp.Eq((x - point2[0])**2 + (y - point2[1])**2, vector_list[1]**2)
    sp.pprint(eq1)
    sp.pprint(eq2)
    (r1, r2) = sp.solve((eq1, eq2), (x, y))
    return (r1, r2)

def interS_l(point1, r, l):
    x, y = sp.symbols('x y', real=True)
    eq1 = sp.Eq((x - point1[0])**2 + (y - point1[1])**2, r**2)
    eq2 = sp.Eq(x-l, 0)
    sp.pprint(eq1)
    sp.pprint(eq2)
    (r1, r2) = sp.solve((eq1, eq2), (x, y))
    return (r1, r2)

# def interS_Sympy(point1, point2, vector_list):
#     C1 = sp.Circle(sp.Point2D(point1[0], point1[1]), vector_list[0])
#     C2 = sp.Circle(sp.Point2D(point2[0], point2[1]), vector_list[1])
#     inters = sp.intersection(C1, C2)
#     sp.pprint(C1)
#     sp.pprint(C2)
#     sp.pprint(inters)

#найдём А как вращение вокруг О1
# vA = sp.Matrix([params['O1A']*np.cos(params['q']), params['O1A']*np.sin(params['q'])])
vA = np.asarray(params['O1A']*sp.Point2D([sp.cos(params['wO1A']*t+params['q']), sp.sin(params['wO1A']*t+params['q'])]))

#B - как пересениче окружности радиуса AB(центр А) и окружности радиуса O2B(центра О2)

B = np.asarray(c_intersection(vA, params['AB'], O2, params['O2B'])[0])
C = np.asarray(c_intersection(B, params['BC'], vA, params['CD']/3)[1])
#координату x для D выразим из окружности с центром в С и радиусом CD
Dy = params['d']
deltaDy = Dy - C[1]
Dx = C[0] + sp.sqrt(params['CD']**2 - deltaDy**2)

D = [Dx, Dy]

# E = (c_intersection(D, params['CD']-params['CE'], [C[1].x, C[1].y], params['CE']))
E = params['CE'] / params['CD'] * (D - C) + C
F = np.asarray(c_intersection(E, params['EF'], O3, params['O3F'])[1])
K = (c_intersection(O1, params['O1A'], O2, params['O2B'] + params['AB']))
# print(B)
# O2 = sp.Matrix([params['a'], -params['c']])
eqK = sp.Eq(vA[0], K[0].x.evalf())
eqK2 = sp.Eq(vA[1], K[0].y.evalf())

eqK3 = sp.Eq(vA[0], K[1].x.evalf())
eqK4 = sp.Eq(vA[1], K[1].y.evalf())
Kvalue = sp.solve(eqK, t)
Kvalue2 = sp.solve(eqK2, t)
Kvalue3 = sp.solve(eqK3, t)
Kvalue4 = sp.solve(eqK4, t)
sp.pprint(Kvalue)
sp.pprint(Kvalue2)
sp.pprint(Kvalue3)
sp.pprint(Kvalue4)
for i in K:
    sp.pprint(i.x.evalf())
    sp.pprint(i.y.evalf())
# Pa = [[vA[0], vA[1]], [B[0].x, B[0].y], [C[1].x, C[1].y], [E[0].x, E[0].y], [F[1].x, F[1].y]]
Pa = [[vA[0], vA[1]], [B[0], B[1]], [C[0], C[1]], [E[0], E[1]], [F[0], F[1]]]


VPa = [[sp.diff(i[0]), sp.diff(i[1])] for i in Pa] 
APa = [[sp.diff(i[0]), sp.diff(i[1])] for i in VPa[:2]] 
VD = [sp.diff(D[0], t), sp.diff(D[1], t)]
print(APa)
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.axis("on")
ax.grid()

xdata, ydata = [], []
ln, = plt.plot([], [], 'b-')


vectors = []
points  = []
titles  = []
lines   = []

# Animation initialisation:
def init_model():
    ax.set_xlim(-25, 100)
    ax.set_ylim(-40, 60)
    plt.title('Model')
    return ln,

def clear_screen():

    while len(vectors):
        vectors[-1].remove()
        vectors.pop()
    #     for i in vectors[-1]:
    #         for j in i:
    #             j.remove()
    #         i.remove()
    #     vectors.pop()

    
    
    while len(points):
        for item in points[-1]:
            item.remove()
        points.pop()

    while len(lines):
        for item in lines[-1]:
            item.remove()
        lines.pop()

    while len(titles):
        titles[-1].remove()
        titles.pop()

    # while len(ax.text):
    #     ax.text[-1].remove()
    #     ax.text.pop()

    while len(ax.patches):
        ax.patches[-1].remove()
        # ax.patches.pop()

    # while len(ax.collections):
    #     ax.collections.remove()
        # if len(ax.collections):
            # ax.collections.pop()


# Animation update on each frame:
def update_model(frame):
    # PFabc = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in Pabc]
    # poly = pltp.Polygon([10, 20], [20, 50], fill=False, edgecolor="blue", linewidth=0.5)
    # ax.patches(poly)
    # plt.plot([vA[0], 61], [vA[1], -1], linestyle="dashed", linewidth=0.5, color="green")
    # plt.plot([40, 61], [0, -1], linestyle="dashed", linewidth=0.5, color="green")
    
    # try:
    clear_screen()
    # FAp = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in Pa]
    FAp = []
    for i in Pa:
        x_val = i[0].subs(t, frame).evalf()
        y_val = i[1].subs(t, frame).evalf()
        if not x_val.is_real or not y_val.is_real:
            return ln,
        FAp.append([float(x_val), float(y_val)])
    # if (K[0].x.subs(t, frame).evalf()== FAp[0]):
        # sp.pprint(t)
    FVp = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in VPa]
    
    Dp = [float(D[0].subs(t, frame).evalf()), float(D[1])]
    # FAp.append(Dp)
    VDp = [float(VD[0].subs(t, frame).evalf()), float(VD[1])]
    # FVp.append(VDp)
    FAAp = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in APa]
    
    # Kp = [float(K[1].x.subs(t, frame).evalf()), float(K[1].y.subs(t, frame).evalf())]
    # at = a.dot(V)/V.norm() * V/V.norm()
    # TAA = [FVp[i]/np.linalg.norm(FVp[i]) for i in range(2)]
    
    TAAp = [FVp[i]/np.linalg.norm(FVp[i]) * (np.dot(FAAp[i], FVp[i])/np.linalg.norm(FVp[i])) for i in range(2)]
    TNAAp = [(np.asarray(FAAp[i]) - np.asarray(TAAp[i])) for i in range(2)]
    print(FAAp, FVp, TAAp)
    points.extend(
        plt.plot(i[0], i[1], marker="o", markersize=4, markeredgecolor="black", markerfacecolor="black")
        for i in [FAp[0], FAp[1], FAp[2], FAp[3], FAp[4], Dp, O1, O2, O3]
    )

    points.extend([
        plt.plot(K[0].x, K[0].y, marker="o", markersize=4, markeredgecolor="black", markerfacecolor="black"),
        plt.plot(K[1].x, K[1].y, marker="o", markersize=4, markeredgecolor="black", markerfacecolor="black"),
    ])
    # points.extend(
    #     plt.plot(Dp[0], 1, marker="o", markersize=4, markeredgecolor="black", markerfacecolor="black")
    # )
    plot_rect(Dp)
    lines.extend([
        plt.plot([O1[0], FAp[0][0]], [O1[1], FAp[0][1]], linewidth=2, color="green"),
        plt.plot([O2[0], FAp[1][0]], [O2[1], FAp[1][1]], linewidth=2, color="green"),
        plt.plot([FAp[2][0], Dp[0]], [FAp[2][1], Dp[1]], linewidth=2, color="green"),
        plt.plot([FAp[3][0], FAp[4][0]], [FAp[3][1], FAp[4][1]], linewidth=2, color="green"),
        plt.plot([O3[0], FAp[4][0]], [O3[1], FAp[4][1]], linewidth=2, color="green")
    ])

    plot_poly([FAp[0], FAp[1], FAp[2]])
    # scale_v = 5
    # EVp3_s = [FVp[3][0], FVp[3][1]*scale_v]
    vectors.extend([
        plt.quiver([i[0]], [i[1]], [j[0]], [j[1]], color="black", units='xy', scale = 1, scale_units='xy', angles='xy')
        for i, j in zip(FAp, FVp)
    ])

    
    # vectors.extend([
    #     plt.quiver([FAp[3][0]], [FAp[3][1]], [FVp[3][0]], [FVp[3][1]], color="black", units='xy', scale = 1, scale_units='xy', angles='xy')
    #     # for i, j in [FAp[3], EVp3_s]
    # ])

    # vectors.extend([
    #     plt.quiver([FAp[4][0]], [FAp[4][1]], [FVp[4][0]], [FVp[4][1]], color="black", units='xy', scale = 1, scale_units='xy', angles='xy')
    #     # for i, j in [FAp[3], EVp3_s]
    # ])

    vectors.extend([
        plt.quiver([Dp[0]], [Dp[1]], [VDp[0]], [VDp[1]], color="black", units='xy', scale = 1, scale_units='xy', angles='xy')
        # for i, j in zip(FAp, FVp)
    ])
    
    # vectors.extend([
    #     plt.quiver([FAp[0][0]], [FAp[0][1]], [FAAp[0][0]], [FAAp[0][1]], color="red", units='xy', scale = 0.5, scale_units='xy', angles='xy')
    #     # for i, j in zip(FAp, FAAp)
    # ])

    vectors.extend([
        plt.quiver([i[0]], [i[1]], [j[0]], [j[1]], color="blue", units='xy', scale = 1, scale_units='xy', angles='xy')
        for i, j in zip(FAp[:2], TAAp)
    ])

    # vectors.extend([
    #     plt.quiver([i[0]], [i[1]], [j[0]], [j[1]], color="orange", units='xy', scale = 1.5, scale_units='xy', angles='xy')
    #     for i, j in zip(FAp[:2], TNAAp)
    # ])
# except (TypeError, ValueError):
    # None
#     # Add titles to points on the plot.
#     titles.extend([
#         ax.text(0, 0, frame, fontsize=10)
#     ])


    return ln,

#Draw a triangle at a given point.
def plot_poly(vt):
    triangle = pltp.Polygon(vt, closed=True, edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(triangle)

def plot_rect(vt):
    h = 2
    w = 5
    rect = pltp.Polygon([[vt[0] - w, vt[1] - h], [vt[0] + w, vt[1] - h], [vt[0] + w, vt[1] + h], [vt[0] - w, vt[1] + h]], closed=True, fill=False, edgecolor="blue", linewidth=1)
    ax.add_patch(rect)

anim = FuncAnimation(fig, update_model, frames=np.linspace(-1.17, 0.481258967392588, 120),
# anim = FuncAnimation(fig, update_model, frames=np.linspace(1.17846591985136, 0.915929182541836, 120),
                        init_func=init_model, blit=True)


from IPython.display import HTML, display
display(HTML(anim.to_jshtml()))
plt.close(fig)



