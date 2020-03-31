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
pdf = np.array([5, 3, 2])
pdf = pdf / float(pdf.sum())
r, rep = random_gen(pdf.cumsum(), population)
pop_by_age = np.c_[rep]


# 1. inventar una PDF cualquiera para la distrib de infectados inicial
S_init_by_age = np.c_[[[3],[20],[1]]]
 

I0 = S_init_by_age
S0 = pop_by_age - I0
E0 = np.zeros([Nages,1])
R0 = np.zeros([Nages,1])

S = S0
E = E0
I = I0
R = R0

R_0 = 2.2

beta = 0.7
betas = np.c_[[[beta],[beta],[beta]]]

sigma = 0.05
sigmas = np.c_[[[sigma],[sigma],[sigma]]]

gamma = beta/R_0
gammas = np.c_[[[gamma],[gamma],[gamma]]]


ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.
t = 0.
time_steps = 0
t_max = 140

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

    # al actualziar S usar el I por edades y el S total.
    Sf = S[:,-1].reshape(3,1)
    St = np.c_[([S[:,-1].sum()]*3)]
    If = I[:,-1].reshape(3,1)
    dS = - St * If / population * betas

    n_S = Sf + dS

    # (( E ))

    It = np.c_[([I[:,-1].sum()]*3)]
    Ef = E[:,-1].reshape(3,1) 
    dE = St * It / population * betas - sigmas * Ef
    n_E = Ef + dE

    # (( I ))

    dI =  sigmas*Ef  - gammas * If
    n_I = If + dI

    # (( R ))

    Rf = R[:,-1].reshape(3,1) 
    dR =  sigmas*Ef  - gammas * If
    n_R = Rf + dR

    S = np.insert(S, [time_steps], n_S, axis=1)
    E = np.insert(E, [time_steps], n_E, axis=1)
    I = np.insert(I, [time_steps], n_I, axis=1)
    R = np.insert(R, [time_steps], n_R, axis=1)

##}}}
#
#
#
#
##------------------------------------------------------- PLOT
##{{{
#
ics = [S[0], S[1], S[2], E[0], E[1], E[2], I[0], I[1], I[2], R[0], R[1], R[2]]
labels = ['S', 'E', 'I', 'R']
labels = ['S[0]', 'S[1]', 'S[2]', 'E[0]', 'E[1]', 'E[2]', 'I[0]',
        'I[1]', 'I[2]', 'R[0]', 'R[1]', 'R[2]'] 
clrs = ['red']*3 + ['blue']*3 + ['green']*3 + ['orange']*3 
t = ts

plt.rcParams['savefig.facecolor'] = "0.8"
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#--- SIMU linear
for i, ic in enumerate(ics):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=1, ax=ax[0],
            label=labels[i], color=clrs[i])
    #sns.scatterplot(t, ic, ax=ax[0])

ax[0].set_xlabel('Time [days]', fontsize=22)
ax[0].set_ylabel('Number infected', fontsize=22)
ax[0].legend()
ax[0].grid()
ax[0].set_title('Simulation')
#---
ax[1].set(yscale="log")
ax[1].yaxis.set_major_formatter(\
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax[1].set_title('Simulation')

#--- SIMU log
for i, ic in enumerate(ics):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=1, ax=ax[1],
            color=clrs[i])
    #sns.scatterplot(t, ic, ax=ax[1])
ax[1].set_xlabel('Time [days]', fontsize=22)
ax[1].set_ylabel('Number infected', fontsize=22)
ax[1].grid()



#--- plt

plt.xticks(rotation=0, fontsize=22)
plt.yticks(rotation=90, fontsize=22)
plt.tight_layout()
fig.savefig('../plt/plot_sim_dists.png')
plt.close()

#}}}
