from manim import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

collision = False

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
        M = [(O[0] + R) + R*sp.cos( sp.pi - qM), O[1] + R*sp.sin(sp.pi - qM)]
    else:
        M = A

    wO = [sp.diff(O[0]), sp.diff(O[1])]
    VrM = [sp.diff(R*sp.cos( sp.pi - qM)), sp.diff(R*sp.sin(sp.pi - qM))]
    VaM = [wO[0] + VrM[0], wO[1] + VrM[1]]

    def count(x):


        rez = pd.Series(
        {
            't'   : t,  
            'R'   : 18,
            'O1O' : O1A,
            'tA'  : tA, 
            'O(t)' : np.array([float(O[0].subs(t,x).evalf()), float(O[1].subs(t,x).evalf()), 0]),
            'A(t)' : np.array([float(A[0].subs(t,x).evalf()), float(A[1].subs(t,x).evalf())]),
            'M(t)' : np.array([float(M[0].subs(t,x).evalf()), float(M[1].subs(t,x).evalf())]),
            'VeO'  : np.array([float(wO[0].subs(t,x).evalf()), float(wO[1].subs(t,x).evalf()), 0])*0.25,
            'VrM'  : np.array([float(VrM[0].subs(t,x).evalf()), float(VrM[1].subs(t,x).evalf()), 0])*0.25,
            'VM'  : np.array([float(VaM[0].subs(t,x).evalf()), float(VaM[1].subs(t,x).evalf()), 0])*0.25,
        }
        )

        global collision
        if (np.linalg.norm([rez['M(t)'][0] - rez['A(t)'][0], rez['M(t)'][1] - rez['A(t)'][1]]) < 0.1):
            collision = True

        if (rez['O(t)'][1] < 0):
            rez['O(t)'][1] = np.abs(rez['O(t)'][1])
        if (rez['A(t)'][1] < 0):
            rez['A(t)'][1] = np.abs(rez['A(t)'][1])
        print(rez['VrM'])
        return (rez)
    return count(x)

class Task(Scene):
    def construct(self):
        # time tracker
        timeTr = ValueTracker(0)
        time = lambda:float(timeTr.get_value())

        

        values = w5_5(0)

        duration = values['tA']
        #Add coordinates.
        axes = Axes(
            x_range=(-20, 40, 10), y_range=(-20, 40, 10),
            tips = True,
            x_length=7, y_length=7,
            axis_config = {"include_numbers": True, "tick_size":0.06,"stroke_width":2, "tip_width": 0.1, "tip_height": 0.1,
                        "font_size":22,"line_to_number_buff":0.1, "stroke_opacity":0.65},
            x_axis_config = {"color":RED_B},
            y_axis_config = {"color":GREEN_B},
            ).shift(RIGHT*1.5)  

        labels = VGroup( axes.get_x_axis_label(MathTex('x', color=RED, font_size=26)),
                    axes.get_y_axis_label(MathTex('y', color=GREEN, font_size=26)))

        def update_A(mob, alpha, point):
            # t = interpolate(0, 1, alpha)
            t = alpha * duration
            values = w5_5(t)[str(point)]
            mob.move_to(axes.c2p(values[0], values[1]))
        
        
        points_coordinats = [(0, 0), (2*values['R'], 0)]
        dots = VGroup(*[Dot(axes.c2p(x, y), color=YELLOW) for x, y in points_coordinats])

        dotA = Dot(axes.c2p(values['O(t)'][0], values['O(t)'][1]), color=ORANGE, radius=0.1)
        dotB = Dot(axes.c2p(values['A(t)'][0], values['A(t)'][1]),color=RED, radius=0.1)

        dotM = Dot(axes.c2p(values['M(t)'][0], values['M(t)'][1]),color=YELLOW, radius=0.1)
        
        dotVeO = Dot(axes.c2p(values['VeO'][0], values['VeO'][1]),color=YELLOW, radius=0)
        dotVrM = Dot(axes.c2p(*values['VrM']),color=YELLOW, radius=0)
        dotVM = Dot(axes.c2p(*values['VM']),color=YELLOW, radius=0)

        drawM = True

                    
        lineA = always_redraw(lambda: Line(start=dots[0], end=dotA.get_center(), color=BLUE))

        lineB = always_redraw(lambda: Line(start=dots[1], end=dotB.get_center(), color=BLUE))
        lineAB = always_redraw(lambda: Line(start=dotA.get_center(), end=dotB.get_center(), color=GREEN))
        
        semicircle = always_redraw(lambda: ArcBetweenPoints(
            start=dotA.get_center(),
            end=dotB.get_center(),
            angle=-PI,
            color=BLUE
        ))
        
        time_line = Line(ORIGIN, RIGHT)
        time_line_dot = Dot(radius=0.04).set_color(ORANGE).move_to(time_line.get_start())
        curent_time_line = VMobject()
        curent_time_line.add_updater(lambda x: x.become(Line(time_line.get_start(), time_line_dot.get_center(), color = ORANGE)))
        time_text = MathTex('t=', font_size= 22)
        time_text.add_updater(lambda x: x.next_to(time_line_dot, DOWN, buff=0.1))
        time_line_num = DecimalNumber( time(), num_decimal_places=3, font_size=18)
        time_line_num.add_updater(lambda d: d.next_to(time_text, RIGHT, buff=0.1))
        time_line_num.add_updater(lambda d: d.set_value(time()) )
        time_line_group = Group(time_line, curent_time_line, time_line_dot, time_text, time_line_num).shift(LEFT*6 + UP*3.5)
        self.add(time_line_group)

        vector = always_redraw(lambda: Arrow(
            start=dotA.get_center(),
            end=dotVeO.get_center(),
            buff=0,
            color=BLUE, 
        ))
        
        vectorVr = always_redraw(lambda: Arrow(
            start=dotM.get_center(),
            end=dotVrM,
            buff=0,
            color=GREEN, 
        ))

        vectorVa = always_redraw(lambda: Arrow(
            start=dotM.get_center(),
            end=dotVM,
            buff=0,
            color=RED, 
        ))


        self.add(dots, dotA, dotB, dotM, semicircle)
        # self.add(vector, vectorVr, vectorVa)
        if (drawM):
            self.remove(dotM)
        self.add(lineA, lineB, lineAB)
        print('run')
        self.play(Create(vector), Create(vectorVr), Create(vectorVa), Create(axes), Create(dots), Create(lineA), Create(lineB), Create(lineAB), Create(semicircle) ,Write(labels), run_time=0.1, rate_func=linear)
        self.play(UpdateFromAlphaFunc(dotA, lambda mob, alpha: update_A(mob, alpha, 'O(t)')),
                UpdateFromAlphaFunc(dotB, lambda mob, alpha: update_A(mob, alpha, 'A(t)')), 
                UpdateFromAlphaFunc(dotM, lambda mob, alpha: update_A(mob, alpha, 'M(t)')), 
                UpdateFromAlphaFunc(dotVeO, lambda mob, alpha: update_A(mob, alpha, 'VeO')), 
                UpdateFromAlphaFunc(dotVrM, lambda mob, alpha: update_A(mob, alpha, 'VrM')), 
                UpdateFromAlphaFunc(dotVM, lambda mob, alpha: update_A(mob, alpha, 'VM')), 
                MoveAlongPath(time_line_dot, time_line),
                 
                timeTr.animate.set_value(duration), 
                 run_time=duration/0.2, rate_func=linear)
        
        # self.wait(2)
