"""
This script calculates polynomial integrals on finite element domains
"""

import numpy as np



def area_triangle(x,y):
    """ Calculates the area of the triangle with vertices given by input """
    return abs(.5 *(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1])))

def element_funcs(x_arr,y_arr):
    x1 = x_arr[0];
    x2 = x_arr[1];
    x3 = x_arr[2];
    y1 = y_arr[0];
    y2 = y_arr[1];
    y3 = y_arr[2];    
    
    b = np.zeros(3)
    c = np.zeros(3)
    b[0] = (y2 - y3)/( (x1-x2)*(y1-y3) - (x1-x3)*(y1-y2))
    c[0] = (x3 - x2)/( (x1-x2)*(y1-y3) - (x1-x3)*(y1-y2))
    b[1] = (y1 - y3)/( (x2-x1)*(y2-y3) - (x2-x3)*(y2-y1))        
    c[1] = (x3 - x1)/( (x2-x1)*(y2-y3) - (x2-x3)*(y2-y1))        
    b[2] = (y2 - y1)/( (x3-x2)*(y3-y1) - (x3-x1)*(y3-y2))
    c[2] = (x1 - x2)/( (x3-x2)*(y3-y1) - (x3-x1)*(y3-y2))   
    a = np.ones(3) - np.multiply(b,x_arr) - np.multiply(c,y_arr)
    return a,b,c 
 
def element_integrals(x,y):
    """
    Given the three vertexes of the triangular element
    this function calculations the integral of 1,x,y,xy,x^2,y^2,
    x^3,x^2*y,x*y^2, and y^3
    
    """    
    int_arr = np.zeros(10)
    A = area_triangle(x,y)
    #print x,y
    int_arr[0] = A
    int_arr[1] = (A/3.)*(np.sum(x))
    int_arr[2] = (A/3.)*(np.sum(y))
    int_arr[3] = (A/12.)*(np.dot(x,y) + 9*int_arr[1]*int_arr[2]/(A**2))
    int_arr[4] = (A/12.)*(np.dot(x,x) + 9*int_arr[1]**2/(A**2))
    int_arr[5] = (A/12.)*(np.dot(y,y) + 9*int_arr[2]**2/(A**2))
    int_arr[6] = x_cubed_integral(x,A)
    int_arr[7] = x_squared_y_integral(x,y,A)
    int_arr[8] = y_squared_x_integral(x,y,A)
    int_arr[9] = y_cubed_integral(y,A)
    return int_arr   

def x_cubed_integral(x,A):    
    """ Calculates the integral of x^3 on a finite element """
    term1 = x[0]**3 + x[1]**3 + x[2]**3
    term2 = ( (x[0]**2)*x[1] + (x[0]**2)*x[2] + (x[1]**2)*x[0]
             +(x[1]**2)*x[2] + (x[2]**2)*x[0] + (x[2]**2)*x[1])
    term3 = x[0]*x[1]*x[2]
    return (A/10.) * (term1+term2+term3)

def y_cubed_integral(y,A):
    """ Calculates the integral of y^3 on a finite element """
    return x_cubed_integral(y,A)
    
def x_squared_y_integral(x,y,A):
    """ Calculates the integral of x^2*y on a finite element """
    term1 = .1*( (x[0]**2)*y[0] + (x[1]**2)*y[1] + (x[2]**2)*y[2] )
    term2 = (1./15.)*( x[0]*x[1]*y[0] + x[0]*x[2]*y[0] + x[0]*x[1]*y[1]
                      +x[1]*x[2]*y[1] + x[0]*x[2]*y[2] + x[1]*x[2]*y[2])
    term3 = (1./30.)*( (x[0]**2)*y[1] + (x[0]**2)*y[2] + (x[1]**2)*y[0]
                      +(x[1]**2)*y[2] + (x[2]**2)*y[0] + (x[2]**2)*y[1])
    term4 = (1./30.)*(x[0]*x[1]*y[2] + x[0]*x[2]*y[1] + x[1]*x[2]*y[0] )
    return A*(term1 + term2 + term3 + term4)

def y_squared_x_integral(x,y,A):
    """ Calculates the integral of y^2*x on a finite element """
    return x_squared_y_integral(y,x,A)

def potential_integrate(a,b,c,Va,Vb,Vc,node1,node2,int_arr):
    """ Calulates the matrix element of the potential with the basis functions """
    i = node1; j = node2
    t = np.zeros(10)
    t[0] = a[i]*a[j]*Va
    t[1] = a[i]*a[j]*Vb + a[j]*Va*b[i] + a[i]*Va*b[j]
    t[2] = a[i]*a[j]*Vc + a[j]*Va*c[i] + a[i]*Va*c[j]
    t[3] = a[j]*b[i]*Vc + a[j]*Vb*c[i] + a[i]*b[j]*Vc + Va*b[j]*c[i] + a[i]*Vb*c[j]+ Va*b[i]*c[j]
    t[4] = a[j]*b[i]*Vb + a[i]*b[j]*Vb + Va*b[i]*b[j]
    t[5] = a[j]*c[i]*Vc + a[i]*c[j]*Vc + Va*c[i]*c[j]
    t[6] = b[i]*b[j]*Vb
    t[7] = b[i]*b[j]*Vc + b[j]*Vb*c[i] + b[i]*Vb*c[j]
    t[8] = b[i]*c[j]*Vc + Vb*c[i]*c[j] + b[j]*c[i]*Vc
    t[9] = c[i]*c[j]*Vc
    return np.dot(t,int_arr)
  
def Theta_mtx_gen(a,b,c,int_arr):
    """ Calculates the Theta matrix (see pg. 6 of Finite Element Method notes in 
        Remarkable Tablet/app for definition and context). """
    Theta_mtx = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            t = np.zeros(6)
            t[0] = a[i]*a[j]
            t[1] = a[i]*b[j] + b[i]*a[j]
            t[2] = a[i]*c[j] + c[i]*a[j]
            t[3] = b[i]*c[j] + c[i]*b[j]
            t[4] = b[i]*b[j]
            t[5] = c[i]*c[j]
            Theta_mtx[i,j] = np.dot(t,int_arr[:6])
    return Theta_mtx        
  
def Lambda_mtx_gen(a,b,c,int_arr):
    """ Calculates the Lambda matrix (see pg C.3 for definition) for an element """
    Lambda_mtx = np.zeros((3,6))
    for m in range(0,3):
        for n in range(0,6):
            if n == 0: 
                i = 0; j = 0;
            elif n == 1: 
                i = 1; j = 1;
            elif n == 2: 
                i = 2; j = 2;
            elif n == 3: 
                i = 0; j = 1;
            elif n == 4: 
                i = 0; j = 2;
            elif n == 5: 
                i = 1; j = 2;
            t = np.zeros(10)
            A = a[i]*a[j]
            B = a[i]*b[j] + b[i]*a[j]
            C = a[i]*c[j] + c[i]*a[j]
            D = b[i]*c[j] + c[i]*b[j]
            E = b[i]*b[j]
            F = c[i]*c[j]
            
            t[0] = a[m]*A
            t[1] = a[m]*B + b[m]*A
            t[2] = a[m]*C + c[m]*A
            t[3] = a[m]*D + b[m]*C + c[m]*B
            t[4] = a[m]*E + b[m]*B
            t[5] = a[m]*F + c[m]*C
            t[6] = b[m]*E
            t[7] = c[m]*E + b[m]*D
            t[8] = b[m]*F + c[m]*D
            t[9] = c[m]*F            
            
            if i == j:
                Lambda_mtx[m,n] = np.dot(t,int_arr)
            else:
                Lambda_mtx[m,n] = 2.*np.dot(t,int_arr)
    return Lambda_mtx,t,int_arr            

def Lambda_check(a,b,c,x,y,m,i):
        j = i
        t = np.zeros(10)
        A = a[i]*a[j]
        B = a[i]*b[j] + b[i]*a[j]
        C = a[i]*c[j] + c[i]*a[j]
        D = b[i]*c[j] + c[i]*b[j]
        E = b[i]*b[j]
        F = c[i]*c[j]
        
        t[0] = a[m]*A
        t[1] = a[m]*B + b[m]*A
        t[2] = a[m]*C + c[m]*A
        t[3] = a[m]*D + b[m]*C + c[m]*B
        t[4] = a[m]*E + b[m]*B
        t[5] = a[m]*F + c[m]*C
        t[6] = b[m]*E
        t[7] = c[m]*E + b[m]*D
        t[8] = b[m]*F + c[m]*D
        t[9] = c[m]*F

        #x = x_arr[i]; y = y_arr[i]
        r = np.zeros(10)
        r[0] = 1; r[1] = x; r[2] = y; r[3] = x*y
        r[4] = x*x; r[5] = y*y; r[6] = x**3
        r[7] = x**2 * y; r[8] = x * y**2
        r[9] = y**3
        
        return np.around(np.dot(t,r),decimals = 8), np.around(a[m] + b[m]*x + c[m]*y,decimals = 8)
def den_check(a,b,c,x,y,i,j):
    t = np.zeros(6)
    t[0] = a[i]*a[j]
    t[1] = a[i]*b[j] + b[i]*a[j]
    t[2] = a[i]*c[j] + c[i]*a[j]
    t[3] = b[i]*c[j] + c[i]*b[j]
    t[4] = b[i]*b[j]
    t[5] = c[i]*c[j]
    r = np.zeros(6)
    r[0] = 1; r[1] = x; r[2] = y; r[3] = x*y
    r[4] = x*x; r[5] = y*y    
    return  np.around(np.dot(t,r),decimals = 8)    
        
    