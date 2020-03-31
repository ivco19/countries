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


#------------------------------------------------------------
# Implementacion del modelo SIR
#
# no hay retardo, el contagio y los sintomas son instantaneos
# no es estocástico
# toda la poblacion reacciona igual al virus
# es cerrado: S+I+R=N
#------------------------------------------------------------



#___________________________________
# Load settings

conf = cv19.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

#----------------------------------------------- SIMULATED MODEL 
#{{{

c = cv19.InfectionCurve()
p = conf.p
g = cv19.Graph()

beta = 0.2
mu = 0.1

population = 1.e5
N_init = 10

S0 = population-N_init
I0 = N_init
R0 = 0

for node in ['S','I','R']:
    g.add_node(node, 0)

g.set_node('S', S0)
g.set_node('I', I0)
g.set_node('R', R0)

# cumulative time series
S = [g.get_node_value('S')] # Susceptible
I = [g.get_node_value('I')] # Infected
R = [g.get_node_value('R')] # Recovered

ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.

T_IC = int(p.t_incubation/ p.dt)

g.add_edge('S', 'I', nms, [beta, 0])
g.add_edge('I', 'R', nms, [mu, 0])




t = 0.
time_steps = 0

while t < p.t_max:

    time_steps = time_steps + 1

    t_prev = t
    t = t + p.dt
    ts.append(t)

    # (( S ))
    prob_SI = g.get_edge('S', 'I', 'prob') # beta
    lag_SI = g.get_edge('S', 'I', 'lag')
    update_SI = S[-lag_SI] if lag_SI < len(S) else S[-1]

    #dS = - S[-1] * ( I[-1]/population ) * prob_SI
    dS = - prob_SI*S[-1]*I[-1]/population

    n_S = S[-1] + dS*p.dt


    # (( I ))
    prob_IR = g.get_edge('I', 'R', 'prob') # mu
    lag_IR = g.get_edge('I', 'R', 'lag')
    update_IR = I[-lag_IR] if lag_IR < len(I) else I[-1]

    #dI = -dS  - prob_IR * update_IR
    dI = beta*S[-1]*I[-1]/population - mu*I[-1]

    n_I = min(I[-1] + dI*p.dt, p.population)

    # (( R ))
    dR = mu*I[-1]

    n_R = R[-1] + dR*p.dt

    S.append(n_S)
    I.append(n_I)
    R.append(n_R)

#}}}


#----------------------------------------------- ANALYTICAL MODEL
#{{{

beta = 0.2
mu = 0.1

AM = EpiModel()
AM.add_interaction('S', 'I', 'I', beta)
AM.add_spontaneous('I', 'R', mu)


S0 = population-N_init
I0 = N_init
R0 = 0

AM.integrate(365, S=S0, I=I0, R=R0)

Sa = AM.values_['S'].values
Ia = AM.values_['I'].values
Ra = AM.values_['R'].values
Ta = range(len(Sa))
#}}}



#------------------------------------------------------- PLOT
#{{{

ics = [S, I, R]
labels = ['S', 'I', 'R']
t = ts

plt.rcParams['savefig.facecolor'] = "0.8"
fig, ax = plt.subplots(2, 2, figsize=(20, 15))

#--- SIMU linear
for ic, lbl in zip(ics, labels):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=4, ax=ax[0,0], label=lbl)
    sns.scatterplot(t, ic, ax=ax[0,0])

ax[0,0].set_xlabel('Time [days]', fontsize=22)
ax[0,0].set_ylabel('Number infected', fontsize=22)
ax[0,0].legend()
ax[0,0].grid()
#---
ax[0,1].set(yscale="log")
ax[0,1].yaxis.set_major_formatter(\
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

#--- SIMU log
for ic in ics:
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[0,1])
    sns.scatterplot(t, ic, ax=ax[0,1])
ax[0,1].set_xlabel('Time [days]', fontsize=22)
ax[0,1].set_ylabel('Number infected', fontsize=22)
ax[0,1].grid()

#--- ANALYTIC linear
ics = [Sa, Ia, Ra]
labels = ['S', 'I', 'R']
t = Ta

for ic, lbl in zip(ics, labels):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=4, ax=ax[1,0], label=lbl)
    sns.scatterplot(t, ic, ax=ax[1,0])

ax[1,0].set_xlabel('Time [days]', fontsize=22)
ax[1,0].set_ylabel('Number infected', fontsize=22)
ax[1,0].legend()
ax[1,0].grid()

#--- ANALYTIC log
ax[1,1].set(yscale="log")
ax[1,1].yaxis.set_major_formatter(\
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

for ic in ics:
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[1,1])
    sns.scatterplot(t, ic, ax=ax[1,1])
ax[1,1].set_xlabel('Time [days]', fontsize=22)
ax[1,1].set_ylabel('Number infected', fontsize=22)
ax[1,0].legend()
ax[1,1].grid()

#--- plt

plt.xticks(rotation=0, fontsize=22)
plt.yticks(rotation=90, fontsize=22)
plt.tight_layout()
fig.savefig('../plt/plot_sim.png')
plt.close()

#}}}
