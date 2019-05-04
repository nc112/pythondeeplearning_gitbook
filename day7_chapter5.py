import matplotlib.pyplot as plt
import numpy as np

'''
Define derivative
'''
def derivativefunc(f, x):
    h = 1e-4
    return (f(x) - f(x+h)) / h

apple = 100
number = 2
tax = 1.1

#forward flow
step_one = apple * number
step_two = step_one * tax
result = step_two

print(step_one, step_two, result)
#backford flow
f = result
rstep_one = result/f
rstep_two = f/step_one
rstep_three = rstep_two*number

print(rstep_one, rstep_two, rstep_three)
