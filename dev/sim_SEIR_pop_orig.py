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
# Esta es una implementacion del modelo SEIR de desarrollo,
# para que luego reemplace a las funciones del modulo cv19
#
# en la version original:
# no hay retardo, el contagio y los sintomas son instantaneos
# no es estocástico
# toda la poblacion reacciona igual al virus
# es cerrado: S+I+R=N
#
# en esta version:
# hay retardo estadistico
# no es estocástico
# es cerrado: S+I+R=N
#
# Mejoras a realizar:
# - incorporar casos importados: simplemente sumar una Poisson a I
# - incorporar la naturaleza estocástica: simplemente sortear una
#   VA en lugar de hacer el calculo determinista
# - incorporar cuarentenas: hacer que beta sea una función del tiempo
#
# Con eso el modelo tendría todo
#
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
pdf = np.array([5, 3, 2])


Nages = len(pdf)
pdf = pdf / float(pdf.sum())
r, rep = random_gen(pdf.cumsum(), population)

rep = np.array([.8,.2,.1])*population

pop_by_age = np.c_[rep]


# Population has a given age distribution
#---------------------------------------------------
S_init_by_age = np.c_[[[3],[20],[1]]]

# Initialize graph:
#---------------------------------------------------
I0 = S_init_by_age
S0 = pop_by_age - I0
E0 = np.zeros([Nages,1])
R0 = np.zeros([Nages,1])

S, E, I, R = S0, E0, I0, R0

zs = np.zeros([Nages,1])
pops = np.c_[[[population],[population],[population]]]

# transition probabilities may depend on the age:
#----------------------------------------------------
R_0 = 2.2

beta = 0.7
betas = np.c_[[[beta],[beta],[beta]]]

sigma = 0.05
sigmas = np.c_[[[sigma],[sigma],[sigma]]]

gamma = beta/R_0
gammas = np.c_[[[gamma],[gamma],[gamma]]]
#----------------------------------------------------


ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.
t = 0.
time_steps = 0
t_max = 140



while t < t_max:

    time_steps = time_steps + 1
    t_prev = t
    t = t + p.dt
    ts.append(t)

    # (( S )) al actualzar S usar el I por edades y el S total.
    Sf = S[:,-1].reshape(3,1) # distrib. el dia anterior
    St = np.c_[([S[:,-1].sum()]*3)] # total S el dia anterior
    If = I[:,-1].reshape(3,1)
    dS = - St * If / population * betas
    #dS = - Sf * If / pop_by_age * betas
    n_S = np.maximum(Sf + dS, zs)

    # (( E ))
    It = np.c_[([I[:,-1].sum()]*3)]
    Ef = E[:,-1].reshape(3,1) 
    dE = St * It / population * betas - sigmas * Ef
    dE = Sf * It / pop_by_age * betas - sigmas * Ef
    n_E = np.minimum(Ef + dE, pop_by_age)

    # (( I ))
    dI =  sigmas*Ef  - gammas * If
    n_I = np.minimum(If + dI, pops)

    # (( R ))
    Rf = R[:,-1].reshape(3,1) 
    dR =  gammas * If
    n_R = np.minimum(Rf + dR, pop_by_age)

    S = np.insert(S, [time_steps], n_S, axis=1)
    E = np.insert(E, [time_steps], n_E, axis=1)
    I = np.insert(I, [time_steps], n_I, axis=1)
    R = np.insert(R, [time_steps], n_R, axis=1)
##}}}
#
    # para el lag:
    # reemplazar I[:,-1] por I[:,-l:] y pesar por la distribución
    # de tiempos de retardo.


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
fig, ax = plt.subplots(1, 3, figsize=(20, 10))

#--- SIMU linear
for i, ic in enumerate(ics):
    if i%3!=0: continue
    sns.lineplot(x=t, y=ic, sort=False, linewidth=1, ax=ax[0],
            label=labels[i], color=clrs[i])
    #sns.scatterplot(t, ic, ax=ax[0])

ax[0].set_xlabel('Time [days]', fontsize=22)
ax[0].set_ylabel('Number infected', fontsize=22)
ax[0].legend()
ax[0].grid()
ax[0].set_title('Simulation')
#---
for i, ic in enumerate(ics):
    if i%3!=1: continue
    sns.lineplot(x=t, y=ic, sort=False, linewidth=1, ax=ax[1],
            label=labels[i], color=clrs[i])
    #sns.scatterplot(t, ic, ax=ax[0])

ax[1].set_xlabel('Time [days]', fontsize=22)
ax[1].set_ylabel('Number infected', fontsize=22)
ax[1].legend()
ax[1].grid()
ax[1].set_title('Simulation')

#---
for i, ic in enumerate(ics):
    if i%3!=2: continue
    sns.lineplot(x=t, y=ic, sort=False, linewidth=1, ax=ax[2],
            label=labels[i], color=clrs[i])
    #sns.scatterplot(t, ic, ax=ax[0])

ax[2].set_xlabel('Time [days]', fontsize=22)
ax[2].set_ylabel('Number infected', fontsize=22)
ax[2].legend()
ax[2].grid()
ax[2].set_title('Simulation')


#--- plt
plt.xticks(rotation=0, fontsize=22)
plt.yticks(rotation=90, fontsize=22)
plt.tight_layout()
fig.savefig('../plt/plot_sim_dists.png')
plt.close()

#}}}
