# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:31:57 2018

@author: Yanndd
"""
import turtle
import numpy as np
from scipy import interpolate
def drawarrow(x,y,direction):
    if direction==180:
        turtle.begin_fill()
        turtle.goto(x,y-2.5)
        turtle.goto(x-10.0,y)
        turtle.goto(x,y+2.5)
        turtle.goto(x,y)
        turtle.end_fill()
        turtle.goto(x-10,y)
        turtle.seth(180)
    if direction== 270:
        turtle.begin_fill()
        turtle.goto(x+2.5,y)
        turtle.goto(x,y-10)
        turtle.goto(x-2.5,y)
        turtle.goto(x,y)
        turtle.end_fill()
        turtle.goto(x,y-10)
        turtle.seth(270)
    if direction == 0:
        turtle.begin_fill()
        turtle.goto(x,y-2.5)
        turtle.goto(x+10,y)
        turtle.goto(x,y+2.5)
        turtle.goto(x,y)
        turtle.end_fill()
        turtle.goto(x+10,y)
        turtle.seth(0)
    if direction == 90:
        turtle.begin_fill()
        turtle.goto(x+2.5,y)
        turtle.goto(x,y+10)
        turtle.goto(x-2.5,y)
        turtle.goto(x,y)
        turtle.end_fill()
        turtle.goto(x,y+10)
        turtle.seth(90)
def createcoor():
    turtle.fd(300)
    turtle.right(135)
    turtle.begin_fill()
    turtle.fd(10)
    turtle.right(135)
    turtle.fd(14)
    turtle.right(135)
    turtle.fd(10)
    turtle.end_fill()
    turtle.pu()
    turtle.goto(305,0)
    turtle.write("w")
    turtle.goto(0,0)
    turtle.seth(90)
    turtle.pd()
    turtle.fd(300)
    turtle.right(135)
    turtle.begin_fill()
    turtle.fd(10)
    turtle.right(135)
    turtle.fd(14)
    turtle.right(135)
    turtle.fd(10)
    turtle.end_fill()
    turtle.pu()
    turtle.goto(0,305)
    turtle.write("E")
x=np.array([0,47,102,227,178,303])
y=np.array([241,98,247,200,240,210])
f=interpolate.interp1d(x,y,kind=5)

createcoor()
turtle.goto(0,0)
turtle.pd()
for i in range(300):
    turtle.goto(i,f(i))

turtle.penup()
turtle.goto(15,60)
turtle.write("globales Minimum")
#turtle.goto(192,112)
#turtle.write("lokales Minimum")
turtle.color("red")
turtle.goto(63,f(63))
turtle.pd()
turtle.seth(180)
turtle.fd(10)
xn,yn=turtle.position()
drawarrow(xn,yn,180)
turtle.left(90)
turtle.goto(43,f(43)+10)
xn,yn=turtle.position()
drawarrow(xn,yn,270)
turtle.seth(180)
turtle.fd(10)
xn,yn=turtle.position()
drawarrow(xn,yn,180)
turtle.seth(90)
turtle.fd(10)
xn,yn=turtle.position()
drawarrow(xn,yn,90)
turtle.seth(0)
turtle.fd(10)
xn,yn=turtle.position()
drawarrow(xn,yn,0)
turtle.pu()
turtle.goto(1000,1000)
turtle.done()
turtle.bye()
