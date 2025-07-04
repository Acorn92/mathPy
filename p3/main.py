import sympy as sp
x = sp.Symbol('x', real=True)

# fy = 5*x**2 - 3*x - sp.exp(x)
# dy = sp.simplify(sp.diff(fy, x))
# sp.pprint(dy)

# fy = x**2 - 3

# sp.plot(fy, autoscale = False, ylim=(-4, 10))
# dy = sp.simplify(sp.diff(fy, x))
# sp.plot(dy, autoscale = False, ylim=(-4, 10))

# y = sp.cos(x)/sp.sin(x)

# sp.pprint(sp.simplify(sp.diff(y, x, 2)))

# a, b = sp.symbols('a b')

# y = a*x + b

# sp.pprint(sp.simplify(sp.diff(y, x, 2)))

# y = x**2 - 3

# dy = sp.integrate(y, x)

# sp.pprint(dy)
# sp.plot(dy, autoscale = False, ylim=(-4, 10))

# y  = ((sp.Pow(x,4) + sp.exp(x)*x**2 - x**2 - 2*x*sp.exp(x) - 2*x - sp.exp(x))*sp.exp(x))/ \
#         (((x-1)**2)*((x+1)**2)*(sp.exp(x)+1))

# sp.pprint(sp.integrate(y, x))

# y = 2*sp.sin(x+sp.pi/4)

# sp.pprint(sp.integrate(y, (x, 0, sp.pi)))

# x,y = sp.symbols('x y')

# f1 = -2*x**2 + 5*x*y - 2*x - 3*y**2 + y + 2
# f2 = x**2 - 2*x*y - x + y**2 + y + 1

# F = sp.Matrix([f1, f2])

# sp.pprint(F.jacobian([x,y]))

###
# Видео 3.6
###


# eqn = sp.Eq(sp.Derivative(y(t), t, 3), (4*sp.Derivative(y(t), t) - 3*sp.Derivative(y(t), t, 2)))
# sp.pprint(eqn)

# res = sp.dsolve(eqn, y(t), ics={y(t).diff(t, 2).subs(t, 0): 2, y(t).diff(t).subs(t, 0): 0, y(0): -1})
# sp.pprint(res)


y = sp.Function('y')
t = sp.symbols('t')
u = sp.Function('u')
u2 = sp.Function('u2')
u3 = sp.Function('u3')

eqn1 = sp.Eq(u(t), sp.Derivative(y(t), t))
sp.pprint(eqn1)
eqn2 = sp.Eq(u2(t), sp.Derivative(u(t), t))
sp.pprint(eqn2)
eqn3 = sp.Eq(u3(t),-2*u2(t) + 3*u(t))
sp.pprint(eqn3)
res = sp.dsolve([eqn1, eqn2, eqn3], [y(t), u(t), u2(t)], ics={.subs(t,0): 3, y(t).diff(y(t), t).subs(t,0): 0, y(0): 1})
sp.pprint(res)