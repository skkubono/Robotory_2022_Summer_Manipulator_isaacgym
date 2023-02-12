import numpy as np
from math import cos, sin
from isaacgym import gymapi

# input = current joint angle qc
# output = current transform matrix Tc, Current position xc
def kinematic_func(qc):
    xc = [0,0,0]
    # Tc = np.array()
    s1 = sin(qc[0])
    c1 = cos(qc[0])
    s2 = sin(qc[1])
    c2 = cos(qc[1])
    s3 = sin(qc[2])
    c3 = cos(qc[2])
    s4 = sin(qc[3])
    c4 = cos(qc[3])
    s5 = sin(qc[4])
    c5 = cos(qc[4])
    s6 = sin(qc[5])
    c6 = cos(qc[5])

    d1 = 0.1273
    a2 = -0.612
    a3 = -0.5723
    d2 = 0.2
    d4 = 0.163941
    d5 = 0.1157
    d6 = 0.0922


    s23 = sin(qc[1] + qc[2])
    c23 = cos(qc[1] + qc[2])
    
    s34 = sin(qc[2] + qc[3])
    c34 = cos(qc[2] + qc[3])     
        
    s234 = sin(qc[1] + qc[2] + qc[3])
    c234 = cos(qc[1] + qc[2] + qc[3])
    
    # Tc(0,0) = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0)
    # Tc(1,0) = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0))
    # Tc(2,0) = -((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6)
    # Tc(3,0) = 0
    
    # Tc(0,1) = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0))
    # Tc(1,1) = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0))
    # Tc(2,1) = -(s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0)
    # Tc(3,1) = 0
    
    # Tc(0,2) = -(((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0)
    # Tc(1,2) = -(c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0)
    # Tc(2,2) = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0)
    # Tc(3,2) = 0
    
    # Tc(0,3) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3)
    # Tc(1,3) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3)
    # Tc(2,3) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234)
    # Tc(3,3) = 1
    
    # xc(0) = Tc(0,3)
    # xc(1) = Tc(1,3)
    # xc(2) = Tc(2,3)

    xc[0] = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3)
    xc[1] = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3)
    xc[2] = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234)


    return xc