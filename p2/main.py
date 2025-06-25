import sympy as sp

# x, y, z, d = sp.symbols('x y z d')
# a1 = sp.Matrix([[6, 2, 0, -1], [1, 3, 2, 7], [5, -2, 7, -4], [8, 1, 3, -4]])
# b1 = sp.Matrix([0, -2, 2, 1])

# rez = sp.linsolve((a1,b1),[x,y,z,d])

# print(rez)

# a = sp.Matrix([[11,0], [33, 0]]).T
# r = a.rref()

# print(r)

# a = sp.Matrix([[1,4], [2,7], [3, 5]])

# cs = a.columnspace()
# print(cs)

# rs = a.rowspace()
# print(rs)

# l, b = sp.symbols('L B')

# a = sp.Matrix([[l, 7, 5, 10], [3, 1, b, 4], [2, 1, 1, 2], [8, 5, 3, 6]])

# r = sp.Matrix.echelon_form(a)
# print(r)

# c1 = sp.Matrix([[1,1,0],[0, 0, 1]]).T
# r1 = sp.Matrix([[1,2], [2,5]])
# # c2 = sp.Matrix([0,0,1])

# # r2 = sp.Matrix([2,5]).T
# sp.pprint(c1)
# sp.pprint(r1)
# a = c1 * r1
# # sp.
# sp.pprint(a)

# cs =sp.Matrix.columnspace(a)
# print(cs)
# k = a.rowspace()
# print(k)

# A = sp.Matrix([[1, -2, -1], [1, -1, 2]])
# b = sp.Matrix([8,2])

# print(A.rank())
# print(A.rref()[0])
# An = A.nullspace()
# print(An)
# res = sp.linsolve((A,b))
# print(res)

# print(res.args[0].subs('tau0', 1))
# rM = sp.Matrix(res.args[0].subs('tau0', 0)) + A.nullspace()[0]*sp.Symbol('tau0')
# print(An[0]*b.T)

# A = sp.Matrix([[-1, 1, 1, -1], [1, 1, 1, -1]]).T

# sp.pprint(A)

# Ac = sp.Matrix.hstack(*(A.columnspace()))
# sp.pprint(Ac)

# P = ((A*(sp.Matrix.inv(A.T*A)))*A.T)
# sp.pprint(P)
# b= sp.Matrix([2,1,-3, 1])

# ap = P * b
# sp.pprint(ap)

# aI = sp.Matrix.eye(4) * b
# sp.pprint(aI)

# rI = sp.Matrix.eye(4) - P
# sp.pprint(rI)

# oP = rI*b
# sp.pprint(oP)

# t = sp.Matrix([[1,-1,1], [1,1,1], [1,-1,1], [4,2,1], [0,0,1], [1, 1, 1]])
# y = sp.Matrix([2, 0, 0, 1, -1, 2])
# rez = t.solve_least_squares(y, method="PINV")
# sp.pprint(rez)
# error2 = (t*rez - y).norm().n(2)
# print(error2)

# p = sp.Matrix([[1, -1], [1,1], [1,-1], [1,2], [1,0], [1,1]])
# rezP = p.solve_least_squares(y, method="PINV")
# sp.pprint(rezP)
# error = (p*rezP - y).norm().n(2)
# print(error)
# import math
# from sympy import Quaternion
# r = sp.Matrix([[0.837, -0.224, 0.5], [0.525, 0.592, -0.612], [-0.159, 0.775, 0.612]])
# sp.pprint(r)
# e = Quaternion.from_rotation_matrix(r).to_euler('zyx')
# e = sp.Matrix(e)
# for i in range(e.rows):
#     e[i] = math.degrees(e[i])
# sp.pprint(e)
# q = Quaternion.from_rotation_matrix(r)
# sp.pprint(q)
# aa = q.to_axis_angle()
# sp.pprint(aa)

# def rotate(angle, o):
#     rez = sp.Matrix()
#     if o == 'x':
#         rez = sp.Matrix([[1, 0, 0, 0],
#                          [0, sp.cos(angle), -sp.sin(angle), 0],
#                          [0, sp.sin(angle), sp.cos(angle),0],
#                          [0, 0, 0, 1]])
#     elif o == 'y':
#         rez = sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), 0],
#                          [0, 1, 0, 0],
#                          [-sp.sin(angle), 0, sp.cos(angle), 0],
#                          [0, 0, 0, 1]])
#     elif o == 'z':
#         rez = sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0, 0],
#                          [sp.sin(angle), sp.cos(angle), 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]])
#     return rez

# def shift(shift, o):
#     rez = sp.Matrix()
#     if o == 'x':
#         rez = sp.Matrix([[1, 0, 0, shift],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#     elif o == 'y':
#         rez = sp.Matrix([[1, 0, 0, 0],[0, 1, 0, shift], [0, 0, 1, 0], [0, 0, 0, 1]])
#     elif o == 'z':
#         rez = sp.Matrix([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, shift], [0, 0, 0, 1]])
#     return rez

# # a1,a2, an1, an2 = sp.symbols('a1 a2 theta1 theta2')

# # H = rotate(an1, 'z')*shift(a1,'x')*rotate(an2, 'z')*shift(a2,'x')        
# # H.simplify()
# # sp.pprint(H)

# l1, l2, l3, l4 = sp.symbols('l1 l2 l3 l4')
# q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')

# import math
# H = shift(l1, 'z')*rotate(q1, 'z')*  \
#     shift(l2, 'x') * rotate(q2,'z') *\
#     shift(l3, 'x')* rotate(q4, 'z') * \
#     shift(-l4, 'z')*rotate(math.radians(-90),'z')* rotate(math.radians(180),'y')

# sp.pprint(H)
# H.simplify()
# sp.pprint(H)

def showImage(srs):
    cv.imshow('', srs)
    cv.waitKey(0)   

def rotate(img, angel, size):
    (h,w) = img.shape[:2]
    center = (int(w/2), int(h/2))
    rotM = cv.getRotationMatrix2D(center, angel, size)
    cos = np.abs(rotM[0,0])
    sin = np.abs(rotM[0,1])

    new_w = int((h*sin) + (w*cos))
    new_h = int((h*cos) + (w*sin))
    rotM[0,2] += (new_w/2) - center[0]
    rotM[1,2] += (new_h/2) - center[1]
    imgRot = cv.warpAffine(img, rotM, (new_w, new_h))
    # gray = cv.cvtColor(imgRot, cv.COLOR_BGR2GRAY)
    gray = imgRot.copy()
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        x, y, wN, hN = cv.boundingRect(cnt)
        cropped = imgRot[y:y+hN, x:x+wN]
        return cropped
    return imgRot

def crop(img, crop_size):
    crop_h = crop_w = crop_size
    x = int((img.shape[1] - crop_w)/2)
    y = int((img.shape[0] - crop_w)/2)
    imgCrop = img[y:y + crop_h, x:x+crop_w]
    return imgCrop

import numpy as np
import cv2 as cv
# from google.colab.patches import cv2_imshow
srcTri = np.array([[0,0], [1,1], [1,0]]).astype(np.float32)
dstTri = np.array([[0,0], [1,2], [1,1]]).astype(np.float32)

warp = cv.getAffineTransform(srcTri, dstTri)
print(warp)
imagePath = r"/home/alx/Projects/MathPy/p2/triangle.png"
srs = cv.imread(imagePath,0)
W = srs.shape[0] 
srs = cv.resize(srs, (int(srs.shape[0]*1.4), int(srs.shape[1]*0.7)), cv.INTER_NEAREST)
srsM = cv.flip(srs, -1)
srs2 = np.vstack((srsM,srs))
(h,w) = srs2.shape[:2]
srsH = rotate(srs, -90, 1)
srsHRes = cv.resize(srsH, (srsH.shape[0], h), cv.INTER_NEAREST)
 
srsHM = cv.flip(srsH, 1)
srsHMRes = cv.resize(srsHM, (srsHM.shape[0], h), cv.INTER_NEAREST)

srs3 = np.hstack((srs2,srsHRes))
srs4 = np.hstack((srsHMRes,srs3))

h5, w5 = srs4.shape[:2]



srsWhite = cv.resize(srs, (srs4.shape[1], srs4.shape[0]), cv.INTER_NEAREST)
transpM = np.float32([[1,0,-int(srsWhite.shape[1]/3)], [0,1, 0]])
hW, wW = srsWhite.shape[:2]
srsWHiteN = cv.warpAffine(srsWhite, transpM, (srsWhite.shape[1], srsWhite.shape[0]))


srsWHiteNM = cv.flip(srsWHiteN, -1)
srs6 = np.vstack((srs4, srsWHiteN))
srs7 = np.vstack((srsWHiteNM, srs6))
srs8 = rotate(srs7, -45, 1)
srs9 = crop(srs8, W*2)
showImage(srs9)