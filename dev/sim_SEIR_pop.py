import numpy as np
from sys import argv
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from collections import Counter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from EpiModel import *
import cv19

#-------------------------------------------------------------
# Implementacion del modelo SEIR
#
# no hay retardo, el contagio y los sintomas son instantaneos
# no es estocástico
# toda la poblacion reacciona igual al virus
# es cerrado: S+I+R=N
#--------------------------------------------------------------


def random_gen(A, n=1):
    from random import random

    res = []
    rep = [0]*len(A)
    for _ in range(n):

        u = random()

        #msg = 'testing if %f is in interval (%f %f) / index: %i'
        y_old = 0.
        for i, y_new in enumerate(A):
            if u > y_old and u < y_new:
                j = i
                #print(msg % (u, y_old, y_new, i))
                break
            else:
                y_old = y_new

        if j<0 or j>= len(A): 
            print(j, len(A))

        res.append(int(j))
        rep[j] = rep[j] + 1
    return(res, rep) 
 






#___________________________________
# Load settings

conf = cv19.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()


#___________________________________
# parameters

R_0 = 2.2
beta  = 0.7
sigma = 0.05
gamma = beta / R_0

population = 100000
N_init = 10
t_max = 200


#----------------------------------------------- SIMULATED MODEL 
#{{{

c = cv19.InfectionCurve()
p = conf.p
g = cv19.Graph_nd()



# al ppo. todos en S con la distrib de la poblacion:
# 1. inventar una PDF cualquiera para la distrib de poblacion:
# en este ejemplo hay Nages=3 rangos: joven, adulto, mayor
Nages = 3
pdf = np.array([8, 1, 1])
pdf = pdf / float(pdf.sum())
r, rep = random_gen(pdf.cumsum(), population)
pop_by_age = np.c_[rep]



# 1. inventar una PDF cualquiera para la distrib de infectados inicial

S_init_by_age = np.c_[[[0],[10],[0]]]
 

I0 = S_init_by_age
S0 = pop_by_age - I0
E0 = np.zeros([Nages,1])
R0 = np.zeros([Nages,1])

S = S0
E = E0
I = I0
R = R0

beta = 0.2
beta = np.c_[[[beta],[beta],[beta]]]
print('beta :::::::::')
print(beta)


ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.
t = 0.
time_steps = 0
t_max = 3

print('S :::::::::')
print(S)
print('I :::::::::')
print(I)

while t < t_max:

    time_steps = time_steps + 1

    t_prev = t
    t = t + p.dt
    ts.append(t)

    # (( S ))

    np.c_[S[:,-1]] * np.c_[I[:,-1]]

    #dS = - S[-1] * ( I[-1]/population ) * prob_SI
    #n_S = np.c_[[[time_steps],[time_steps],[time_steps]]]

    #S = np.insert(S, [time_steps], n_S, axis=1)


#    #update_SI = S[-lag_SI] if lag_SI < len(S) else S[-1]
#    #dS = - S[-1] * ( I[-1]/population ) * prob_SI
#
#    dS = - beta*S[-1]*I[-1]/population
#
#    n_S = S[-1] + dS*p.dt
# 
#    # (( E ))
#    #prob_SI = g.get_edge('S', 'I', 'prob') # beta
#    #lag_SI = g.get_edge('S', 'I', 'lag')
#    #update_SI = S[-lag_SI] if lag_SI < len(S) else S[-1]
#    #dS = - S[-1] * ( I[-1]/population ) * prob_SI
#
#    dE = beta*S[-1]*I[-1]/population - sigma*E[-1]
#
#    n_E = E[-1] + dE*p.dt
#
#    # (( I ))
#    #prob_IR = g.get_edge('I', 'R', 'prob') # mu
#    #lag_IR = g.get_edge('I', 'R', 'lag')
#    #update_IR = I[-lag_IR] if lag_IR < len(I) else I[-1]
#    #dI = -dS  - prob_IR * update_IR
#
#    dI =  sigma*E[-1] - gamma*I[-1]
#
#    n_I = I[-1] + dI*p.dt
#
#    # (( R ))
#    dR = gamma*I[-1]
#
#    n_R = R[-1] + dR*p.dt


    # Update S
    # INSERT COLUMN: reemplaza a: S.append(n_S)
    #k = S.shape[1]
    #k = 1
    #n_S = np.c_[[[time_steps],[time_steps],[time_steps]]]
    #S = np.insert(S, [k], n_S, axis=1)


#np.ndarray(shape=(9,1), dtype=float)


#    E.append(n_E)
#    I.append(n_I)
#    R.append(n_R)
#
##}}}
#
#
#
#
##------------------------------------------------------- PLOT
##{{{
#
#ics = [S, E, I, R]
#labels = ['S', 'E', 'I', 'R']
#t = ts
#
#plt.rcParams['savefig.facecolor'] = "0.8"
#fig, ax = plt.subplots(1, 2, figsize=(20, 10))
#
##--- SIMU linear
#for ic, lbl in zip(ics, labels):
#    sns.lineplot(x=t, y=ic, sort=False, linewidth=4, ax=ax[0], label=lbl)
#    sns.scatterplot(t, ic, ax=ax[0])
#
#ax[0].set_xlabel('Time [days]', fontsize=22)
#ax[0].set_ylabel('Number infected', fontsize=22)
#ax[0].legend()
#ax[0].grid()
#ax[0].set_title('Simulation')
##---
#ax[1].set(yscale="log")
#ax[1].yaxis.set_major_formatter(\
#        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
#ax[1].set_title('Simulation')
#
##--- SIMU log
#for ic in ics:
#    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[1])
#    sns.scatterplot(t, ic, ax=ax[1])
#ax[1].set_xlabel('Time [days]', fontsize=22)
#ax[1].set_ylabel('Number infected', fontsize=22)
#ax[1].grid()
#
#
#
##--- plt
#
#plt.xticks(rotation=0, fontsize=22)
#plt.yticks(rotation=90, fontsize=22)
#plt.tight_layout()
#fig.savefig('../plt/plot_sim.png')
#plt.close()
#
#}}}
