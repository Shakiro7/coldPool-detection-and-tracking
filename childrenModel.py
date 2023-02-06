#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random as rn 



n = 100000
rep1 = 30
rep2 = 2*rep1
prce = 0.035
pdiu = 0.08
a0 = 6 # initial area as the median initial area of all LES CPs
l0 = 1.6 # perimeter segment length acc. to rain patch min. area 2 km²

save = True # Export plot if True

# The main function, allowing to simulate the stochastic expansion of a cold pool
# parameters are: 
# p   = probability of a segment to experience a new raincell during one timestep Delta t.
# rep = the number of timesteps Delta t to simulate
# n   = the number of cold pools simulated
# r0  = the initial cold pool redius

def distribution(p, rep, n, a0, l0):
    
    areas = np.ones(n) * a0
    active = np.ones(n)
    
    for r in range(rep):   # repeat rep times
        for i in range(n): # repeat for each CP
            old = areas[i] # determine the current CP area
            if active[i] == 1:
                perimeter = np.sqrt(4 * np.pi * areas[i])
                for re in range(int(perimeter/l0)): # carry out a certain number of microsteps
                    if rn.random() < p:
                        areas[i] = areas[i] + a0
                if old == areas[i]:  # area hasn't changed
                    active[i] = 0    # cold pools whose area doesn't change within Delta t are considered dead.
    
    asort = np.sort(areas/a0)     # the number of "children" equals the total area divided by the area a0
    prob  = 1-np.arange(n)/n         # just a probability array
    return asort, prob 


# Plotting various results on a log-log scale.

# setting the font etc.
# Set text properties
txtsize= '18'
plt.rcParams['font.size'] = txtsize

# making an array for some xvals
xvals = np.arange(1,1000)


# simulating three parameter sets: rce with 30 expansion steps, diu with 30 steps, diu with 60 steps
# the difference between rce and diu is the parameter p (first entry in the function below)
aso1, pr1 = distribution(prce,rep1, n, a0, l0)
aso2, pr2 = distribution(pdiu, rep1, n, a0, l0)
aso3, pr3 = distribution(pdiu, rep2, n, a0, l0)

# just plotting the three.
fig, ax = plt.subplots(figsize=(7,5.25))
plt.plot(aso2, pr2, label = ('p='+ str(pdiu) + ', "diu4K"'), color = "indianred")
plt.plot(aso3, pr3, label = '"longer day"', color = "brown", linewidth = .5)
plt.plot(aso1, pr1, label = ('p='+ str(prce) + ', "rce0K"'), color = "olive")

# plotting some power-law and exponential lines
plt.plot(xvals, xvals**(-1.5),label='$\propto k_c^{-1.5}$',color = 'grey', linestyle = 'dashed')
plt.plot(xvals, np.exp(-xvals),label='$\propto exp(-k_c)$', color = 'grey', linestyle = 'dotted')

plt.xscale("log")
plt.yscale("log")
plt.ylim([0.0001,10**0])
plt.yticks([10**(-4),10**(-2),10**0])  
plt.xlim([0.8, 10**3])
plt.xlabel("Number of children, $k_c$")
plt.ylabel("Exceedence probability, 1 - CDF($k_c$)")
plt.legend()

fig.tight_layout()    
if save:
    plt.savefig("cdfChildrenModel.png",bbox_inches='tight',dpi=300)  
plt.show()



# plt.plot(aso1*a0, pr1, label = ('p='+ str(prce) + ', "rce0K"'), color = "gray")
# plt.plot(aso3*a0, pr3, label = '"longer day"', color = "brown", linewidth = .5)
# plt.plot(aso2*a0, pr2, label = ('p='+ str(pdiu) + ', "diu4K"'), color = "orange")

# # plotting some power-law and exponential lines
# xvals = np.arange(1,10000)
# plt.plot(xvals, 1*xvals**(-1.0),label='$\propto k^{-1.5}$',color = 'orange', linestyle = 'dashed')
# plt.plot(xvals, np.exp(-xvals),label='$\propto exp(-k)$', color = 'gray', linestyle = 'dashed')

# plt.text(5000, 0.0003, '$\propto {A_{max}}^{-1.0}$', fontsize=12)
# plt.text(1.5, 0.0003, '$\propto exp(-A_{max})$', fontsize=12)
# plt.ylim(.0001,1)
# plt.xlim(1,5*10**4)

# # setting double-log scales
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Max cold pool area')
# plt.ylabel('Exceedence probability, 1-CDF(k)')

# # saving the plot as png
# #plt.savefig('cdfChildren_model.png',bbox_inches='tight')
# plt.show()


# We now want to plot on a semi-log scale.

# plt.plot(xvals, xvals**(-1.5),label='$\propto k^{-1.5}$', color = "orange")
# plt.plot(xvals, np.exp(-xvals),label='$\propto exp(-k)$', linestyle = "dashed", color = "gray")
# plt.plot(aso1,pr1,label='Simple model', color = "gray")
# plt.plot(aso2,pr2,label='Simple model', color = "orange")
# #plt.xscale('log')
# plt.yscale('log')
# plt.xlim(.9,20)
# plt.ylim(.0001,1)

# plt.xlabel('Number of children, k')
# plt.ylabel('Exceedence probability, 1-CDF(k)')
# plt.legend()
# plt.show()



