'''
感知机
'''
import numpy as np
parameter = [0.5, 0.5, -0.7, #AND
             -0.5, -0.5, 0.7, #NAND
             0.5, 0.5, -0.2] #OR

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([parameter[0], parameter[1]])
    b = parameter[2]
    Sum = np.sum(w*x) + b
    if Sum <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([parameter[3], parameter[4]])
    b = parameter[5]
    Sum = np.sum(w*x) + b
    if Sum <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([parameter[6], parameter[7]])
    b = parameter[8]
    Sum = np.sum(w*x) + b
    if Sum <= 0:
        return 0
    else:
        return 1

def NOR(x1, x2):
    temp1 = NAND(x1, x2)
    temp2 = OR(x1, x2)
    sum = AND(temp1, temp2)
    if sum <=0:
        return 0
    else:
        return 1

#Validation
if 0 == AND(0,1) and\
   0 == AND(1,0) and\
   0 == AND(0,0) and\
   1 == AND(1,1):
    print('Succeed')
else:
    print('failed with AND')

#Test2
if 1 == NAND(0,1) and\
   1 == NAND(1,0) and\
   1 == NAND(0,0) and\
   0 == NAND(1,1):
    print('Succeed')
else:
    print('failed with NAND')

if 1 == OR(0,1) and\
   1 == OR(1,0) and\
   0 == OR(0,0) and\
   1 == OR(1,1):
    print('Succeed')
else:
    print('failed with OR')

if 1 == NOR(0,1) and\
   1 == NOR(1,0) and\
   0 == NOR(0,0) and\
   0 == NOR(1,1):
    print('Succeed')
else:
    print('failed with NOR')