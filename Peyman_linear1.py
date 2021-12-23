# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:20:11 2021

@author: Peyman
"""

'''part a'''
import numpy as np
x = np.random.uniform(-1, 1, 100)

'''part b'''
np.random.seed(8) #last two digits are 08

'''part c'''
y = 12 * x - 4

'''part d'''
import matplotlib.pyplot as plt
plt.scatter(x, y, alpha = 0.5)
plt.title("part d")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

'''part e'''
noise = np.random.normal(0, 1, 100) #standard normal distribution
y += noise

'''part f'''
import matplotlib.pyplot as plt
plt.scatter(x, y, alpha = 0.5)
plt.title("part f after adding noise to y")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

'''part g'''
#This will be in analysis report