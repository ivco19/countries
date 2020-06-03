import numpy as np
from sys import argv
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats

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
#{{{
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
 
# IDEA:
# r, rep = random_gen(pdf.cumsum(), population)
#}}}









#___________________________________
# Load settings

config = cv19.parser()
config.check_file(argv)
config.read_config_file()
config.load_filenames()
config.load_parameters()
p = config.p

IDM = cv19.IDModel()  # Infectious desease model


#___________________________________
# Experiment configuration

# number of age bins
age_bins = 5

# number of days for time lags
lag_days = 40

# maximum number of days to simulate
t_max = 150


#___________________________________
# parameters

total_population = 1.e4
susceptible_initial = 10

# simular una piramide poblacional
ages_intervals = np.linspace(0, 100, age_bins+1)
ages = (ages_intervals[1:] + ages_intervals[:-1])/2
ages_intervals = [int(i) for i in ages_intervals]

pop = stats.norm.pdf(ages, loc=0, scale=70)
POP_S = pop / pop.sum() * total_population


# simular tiempos de retardo (S -> I)
LAG_SE = stats.norm.pdf(range(-lag_days, 0), loc=-10, scale=3)
LAG_EI = stats.norm.pdf(range(-lag_days, 0), loc=-10, scale=3)
LAG_IR = stats.norm.pdf(range(-lag_days, 0), loc=-10, scale=3)


# simular probabilidades de transicion

R_0 = 2.2
beta  = 0.7
sigma = 0.05
gamma = beta / R_0

TRA_SE = [beta]*age_bins
TRA_EI = [sigma]*age_bins
TRA_IR = [gamma]*age_bins

# prueba en beta
#beta  = 0.4
#TRA_SE = [beta]*age_bins
#TRA_SE[-1] = 0.9

# prueba en sigma
#sigma = 0.05
#TRA_EI = [sigma]*age_bins
#TRA_EI[-1] = 0.2

# prueba en gamma
R_0 = 5.2
gamma = beta / R_0
TRA_IR[-1] = gamma





TRA_SE = np.c_[TRA_SE]
TRA_EI = np.c_[TRA_EI]
TRA_IR = np.c_[TRA_IR]

POP_vector = np.c_[POP_S]


# Initialize graph:
#---------------------------------------------------
I0 = np.random.choice(range(5, 10), [age_bins,1])

I0 = np.c_[[10]*age_bins]


S0 = POP_vector - I0
E0 = np.c_[[100]*age_bins] 
R0 = np.zeros([age_bins,1])

S, E, I, R = S0, E0, I0, R0

zs = np.zeros([age_bins,1])


#----------------------------------------------------


ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.
t = 0.
time_steps = 0
randomize=False
randomize_sd = 0.01

while t < t_max:

    time_steps = time_steps + 1
    t_prev = t
    t = t + p.dt
    ts.append(t)

    # (( S )) al actualzar S usar el I por edades y el S total.
    Sf = S[:,-1].reshape(age_bins,1) # distrib. el dia anterior
    St = np.c_[([S[:,-1].sum()]*age_bins)] # total S el dia anterior
    If = I[:,-1].reshape(age_bins,1)

    # aca implementar la binomial
    increment = - St * If * TRA_SE
    if randomize:
        sd = Sf*randomize_sd
        dS = np.random.normal(loc=increment, scale=sd, size=(age_bins,1))
        dS = dS / total_population
    else:
        dS = increment / total_population
    n_S = np.maximum(Sf + dS, zs)

    # (( E ))
    It = np.c_[([I[:,-1].sum()]*age_bins)]
    Ef = E[:,-1].reshape(age_bins,1) 
    
    increment = Sf * It / POP_vector * TRA_SE - TRA_EI * Ef
    
    if randomize:
        sd = Ef*randomize_sd
        dE = np.random.normal(loc=increment, scale=sd, size=(age_bins,1))
    else:
        dE = increment
    
    n_E = np.minimum(Ef + dE, POP_vector)

    # (( I ))
    increment =  TRA_EI * Ef  - TRA_IR * If

    if randomize:
        sd = If*randomize_sd
        dI = np.random.normal(loc=increment, scale=sd, size=(age_bins,1))
    else:
        dI = increment

    n_I = np.minimum(If + dI, POP_vector)

    # (( R ))
    Rf = R[:,-1].reshape(age_bins,1) 
    increment =  TRA_IR * If

    if randomize:
        sd = Rf*randomize_sd
        dR = np.random.normal(loc=increment, scale=sd, size=(age_bins,1))
    else:
        dR = increment

    n_R = np.minimum(Rf + dR, POP_vector)

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

import itertools as it

ccolors = ['maroon','saddlebrown','chocolate','darkorange','moccasin']

ccolors = ['darkslategrey','darkgreen', 'teal','steelblue','dodgerblue']

from matplotlib import cm
cmap = cm.get_cmap('Blues', age_bins+3)
ccolors = cmap(np.linspace(0, 1, age_bins+3))
ccolors = ccolors[3:]

cstyles = ['-']
cwidths = [1, 1, 1, 1,3]
cwidths = [2]

icolors = it.cycle(ccolors)
istyles = it.cycle(cstyles)
iwidths = it.cycle(cwidths)

t = ts

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams['savefig.facecolor'] = "0.95"
#plt.tight_layout(pad=10.6, w_pad=0.6, h_pad=50.0)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0.15
plt.rcParams['axes.ymargin'] = 0.15
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'    
plt.rcParams['axes.xmargin'] = 0.01
plt.rcParams['axes.ymargin'] = 0.01

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 22
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots(2, 2)

#--- SIMU linear
for i, ic in enumerate(S):
    sns.lineplot(x=t, y=ic, sort=False, ax=ax[0,0],
                 label=f"S for age in ({ages_intervals[i]}, {ages_intervals[i+1]})", 
                 color=next(icolors), linewidth=next(iwidths),
                 linestyle=next(istyles))

ax[0, 0].set_xlabel('Time [days]')
ax[0, 0].set_ylabel('Number')
ax[0, 0].set_xlim(0, 150)
ax[0, 0].set_ylim(0, 2800)
ax[0, 0].legend()
ax[0, 0].set_title('S')
#---
for i, ic in enumerate(E):
    sns.lineplot(x=t, y=ic, sort=False, ax=ax[0,1],
                 label=f"E for age in ({ages_intervals[i]}, {ages_intervals[i+1]})", 
                 color=next(icolors), linewidth=next(iwidths),
                 linestyle=next(istyles))

ax[0, 1].set_xlabel('Time [days]')
ax[0, 1].set_ylabel('Number')
ax[0, 1].set_xlim(0, 150)
ax[0, 1].set_ylim(0, 2800)
ax[0, 1].legend()
ax[0, 1].set_title('E')

#---
for i, ic in enumerate(I):
    sns.lineplot(x=t, y=ic, sort=False, ax=ax[1, 0],
                 label=f"I for age in ({ages_intervals[i]}, {ages_intervals[i+1]})", 
                 color=next(icolors), linewidth=next(iwidths),
                 linestyle=next(istyles))

ax[1, 0].set_xlabel('Time [days]')
ax[1, 0].set_ylabel('Number')
ax[1, 0].set_xlim(0, 150)
ax[1, 0].set_ylim(0, 450)
ax[1, 0].legend()
ax[1, 0].set_title('I')

#---
for i, ic in enumerate(R):
    sns.lineplot(x=t, y=ic, sort=False, ax=ax[1, 1],
                 label=f"R for age in ({ages_intervals[i]}, {ages_intervals[i+1]})", 
                 color=next(icolors), linewidth=next(iwidths),
                 linestyle=next(istyles))

ax[1, 1].set_xlabel('Time [days]')
ax[1, 1].set_ylabel('Number')
ax[1, 1].set_xlim(0, 150)
ax[1, 1].set_ylim(0, 2800)
ax[1, 1].legend()
ax[1, 1].set_title('R')



#--- plt
fig.savefig('plot_compartments.png')
fig.savefig('plot_compartments.pdf')
plt.close()

#}}}
