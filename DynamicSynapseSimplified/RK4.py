# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:32:58 2016

@author: chitianqilin
"""

def rk4(h, y, inputs, f):
        k1 = f(y, inputs)
        k2 = f(y + 0.5*h*k1, inputs)
        k3 = f(y + 0.5*h*k2, inputs)
        k4 = f(y + k3*h, inputs)
        return y + (k1 + 2*(k2 + k3) + k4)*h/6.0
        
        
        
#def rk4(x, h, y, f):
#        k1 = h * f(x, y)
#        k2 = h * f(x + 0.5*h, y + 0.5*k1)
#        k3 = h * f(x + 0.5*h, y + 0.5*k2)
#        k4 = h * f(x + h, y + k3)
#        return x + h, y + (k1 + 2*(k2 + k3) + k4)/6.0