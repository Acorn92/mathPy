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

t = sp.symbols('t')

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
vA = params['O1A']*sp.Point2D([sp.cos(t+params['q']), sp.sin(t+params['q'])])

#B - как пересениче окружности радиуса AB(центр А) и окружности радиуса O2B(центра О2)

B = (c_intersection(vA, params['AB'], O2, params['O2B']))
C = (c_intersection([B[0].x, B[0].y], params['BC'], vA, params['CD']/3))
#координату x для D выразим из окружности с центром в С и радиусом CD
Dy = params['d']
deltaDy = Dy - C[1].y
Dx = C[1].x + sp.sqrt(params['CD']**2 - deltaDy**2)

D = [Dx, Dy]

E = (c_intersection(D, params['CD']-params['CE'], [C[1].x, C[1].y], params['CE']))
F = (c_intersection([E[0].x, E[0].y], params['EF'], O3, params['O3F']))
# print(B)
# O2 = sp.Matrix([params['a'], -params['c']])
# sp.pprint(O2[0])
Pa = [[vA[0], vA[1]], [B[0].x, B[0].y], [C[1].x, C[1].y], [E[0].x, E[0].y], [F[1].x, F[1].y]]
VPa = [[sp.diff(i[0], t), sp.diff(i[1], t)] for i in Pa] 
VD = [sp.diff(D[0], t), sp.diff(D[1], t)]

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
    ax.set_xlim(-100, 105)
    ax.set_ylim(-100, 80)
    plt.title('Model')
    return ln,

def clear_screen():
    while len(ax.patches):
        ax.patches[-1].remove()
        ax.patches.pop()

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

    while len(vectors):
        for item in vectors[-1]:
            vectors[-1].remove()
        vectors.pop()

# Animation update on each frame:
def update_model(frame):
    # PFabc = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in Pabc]
    # poly = pltp.Polygon([10, 20], [20, 50], fill=False, edgecolor="blue", linewidth=0.5)
    # ax.patches(poly)
    # plt.plot([vA[0], 61], [vA[1], -1], linestyle="dashed", linewidth=0.5, color="green")
    # plt.plot([40, 61], [0, -1], linestyle="dashed", linewidth=0.5, color="green")
    clear_screen()
    try:
        FAp = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in Pa]
        FVp = [[float(i[0].subs(t, frame).evalf()), float(i[1].subs(t, frame).evalf())] for i in VPa]
        
        Dp = [float(D[0].subs(t, frame).evalf()), D[1]]
        VDp = [float(VD[0].subs(t, frame).evalf()), VD[1]]
        points.extend(
           plt.plot(i[0], i[1], marker="o", markersize=4, markeredgecolor="black", markerfacecolor="black")
            for i in [FAp[3], FAp[4], O1, O2, O3]
        )
        plot_rect(Dp)
        lines.extend([
            plt.plot([O1[0], FAp[0][0]], [O1[1], FAp[0][1]], linewidth=2, color="green"),
            plt.plot([O2[0], FAp[1][0]], [O2[1], FAp[1][1]], linewidth=2, color="green"),
            plt.plot([FAp[2][0], Dp[0]], [FAp[2][1], Dp[1]], linewidth=2, color="green"),
            plt.plot([FAp[3][0], FAp[4][0]], [FAp[3][1], FAp[4][1]], linewidth=2, color="green"),
            plt.plot([O3[0], FAp[4][0]], [O3[1], FAp[4][1]], linewidth=2, color="green")
        ])

        plot_poly([FAp[0], FAp[1], FAp[2]])

        vectors.extend([
            plt.quiver([i[0]], [i[1]], [j[0]], [j[1]], color="black", units='xy', scale = 1, scale_units='xy', angles='xy')
            for i, j in zip(FAp, FVp)
        ])
        vectors.extend([
            plt.plot([Dp[0]], [Dp[1]], [VDp[0]], [VDp[1]], linewidth=2, color="green")
        ])
    except (TypeError, ValueError):
        None
    

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

anim = FuncAnimation(fig, update_model, frames=np.linspace(0, np.pi*2, 120),
                        init_func=init_model, blit=True)


from IPython.display import HTML, display
display(HTML(anim.to_jshtml()))
plt.close(fig)