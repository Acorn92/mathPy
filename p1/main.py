import sympy as sp
from sympy import MatrixSymbol, Matrix, Symbol, symbols, Eq, solve
import math as mt
# b = sp.Matrix([[4], [3]])
# m = sp.Matrix([[3], [4]])
# # # c = sp.Matrix([[5], [0], [-51]])

# # r = sp.Matrix.dot(a,b)
# # print(r)

# # print(3*(a + b) - 5*c)
# # print(2*)
# # r = (float)(3*sp.sqrt(2)*mt.cos(2.35619))
# # print( r ) 

# # Определяем переменные
# x, y = symbols('x y')

# a = sp.Matrix([[-2], [6]])
# dlA = a.norm()
# # Определяем уравнение
# equation = Eq(sp.sqrt((x - 3)**2 + (y + 2)**2), dlA)
# # Решаем уравнение относительно y
# solution = solve(equation, y)
# print(solution)

# import sympy as sp
# # from sympy import MatrixSymbol, Matrix

# a = MatrixSymbol('a', 3, 1)
# b = MatrixSymbol('b', 3, 1)

# vl = (a + b)
# vp = (a - b)


# r1 = sp.Matrix.cross(Matrix(vl), Matrix(vp))
# print(r1)

# r2 = sp.sympify(r1)
# print(r2)

# import sympy as sp

# a = sp.Matrix([[3, 8], [2,6]])

# m1 = sp.Matrix([9, 1]).T
# m2 = sp.Matrix([[2,5],[1,3]])
# m3 = sp.Matrix([[2,5,8], [1,3,7]])

# m4 = (m1 * m2 * m3)

# m5 = sp.Matrix([2,15])
# m6 = sp.Matrix([3, 8, 1]).T

# m7 = m5 * m6

# m8 = m4*m7
# print(m7)
# print(m4)
# print(m8)


# r = a.inv()

# print(r)

# l = symbols('l')

# a = sp.Matrix([[1 - l, 9], [4, 1 - l]])

# eq = sp.det(a)
# s = solve(eq, l)

# print(s)


# a = sp.Matrix([[2-3*sp.I], [2*sp.I+ 5]]).T
# b = sp.Matrix([[2], [15-5*sp.I]])

# z = a * b
# z1 = sp.sympify(z)
# print(z1)

a = sp.Matrix([[2, 5, 0], [1, 3 - sp.I, 2 + sp.I]])
b= sp.Matrix([[5, sp.I, 0], [1, 14, -2 - 3*sp.I]])

a1 = a.pinv()
x = b*a1
x1 = sp.simplify(x)
print(x1)