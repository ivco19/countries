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


#___________________________________
# Load settings

conf = cv19.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

#___________________________________
# parameters (TESTING: REESCRIBE DEL ARCHIVO SET)

R_0 = 5.2
beta  = 0.7
sigma = 0.05
gamma = beta / R_0

population = 1.e5
N_init = 10
t_max = 200
exposed_init = 1000

#----------------------------------------------- SIMULATED MODEL 
#{{{

c = cv19.InfectionCurve()
p = conf.p
g = cv19.Graph()
S0 = population-N_init
E0 = exposed_init
I0 = N_init
R0 = 0

for node in ['S','E','I','R']:
    g.add_node(node, 0)
g.set_node('S', S0)
g.set_node('E', E0)
g.set_node('I', I0)
g.set_node('R', R0)
S = [g.get_node_value('S')] # Susceptible
E = [g.get_node_value('E')] # Exposed    
I = [g.get_node_value('I')] # Infected
R = [g.get_node_value('R')] # Recovered

ts = [0.] # time series
nms = ['prob','lag']
p_dt = 1.
T_IC = int(p.t_incubation/ p.dt)

g.add_edge('S', 'E', nms, [beta, 0])
g.add_edge('E', 'I', nms, [sigma, 0])
g.add_edge('I', 'R', nms, [gamma, 0])

t = 0.
time_steps = 0
while t < t_max:
    time_steps = time_steps + 1
    t_prev = t
    t = t + p.dt
    ts.append(t)

    # (( S ))
    dS = - beta*S[-1]*I[-1]/population
    n_S = S[-1] + dS*p.dt
    # (( E ))
    dE = beta*S[-1]*I[-1]/population - sigma*E[-1]
    n_E = E[-1] + dE*p.dt
    # (( I ))
    dI =  sigma*E[-1] - gamma*I[-1]
    n_I = I[-1] + dI*p.dt
    # (( R ))
    dR = gamma*I[-1]
    n_R = R[-1] + dR*p.dt

    S.append(n_S)
    E.append(n_E)
    I.append(n_I)
    R.append(n_R)
#}}}


#-------------------------- ANALYTICAL MODEL: EPIDEMIOLOGY 101
#{{{

#beta = 0.2   # infection rate?
#sigma = 0.9  # infection rate?
#gamma = 0.1  # recuperation rate

AM = EpiModel()
#AM.add_interaction('S', 'I', 'I', beta)
#AM.add_spontaneous('E', 'I', sigma)
#AM.add_spontaneous('I', 'R', gamma)
AM.add_interaction('S', 'I', 'I', beta)
AM.add_spontaneous('E', 'I', sigma)
AM.add_spontaneous('I', 'R', gamma)

S0 = population-N_init
E0 = exposed_init
I0 = N_init
R0 = 0

AM.integrate(t_max, S=S0, E=E0, I=I0, R=R0)
Sa = AM.values_['S'].values
Ea = AM.values_['E'].values
Ia = AM.values_['I'].values
Ra = AM.values_['R'].values
Ta = range(len(Sa))
#}}}


# -------------------------- ANALYTICAL MODEL: SEIRSPLUS
#{{{
# from seirsplus.models import *
# import networkx
# numNodes = 10000
# baseGraph = networkx.barabasi_albert_graph(n=numNodes, m=9)
# G_normal = custom_exponential_graph(baseGraph, scale=100)
# # Social distancing interactions:
# G_distancing = custom_exponential_graph(baseGraph, scale=10)
# # Quarantine interactions:
# G_quarantine = custom_exponential_graph(baseGraph, scale=5)
# model = SEIRSNetworkModel(G=G_normal, beta=0.155, sigma=1/5.2,
#                           gamma=1/12.39, mu_I=0.0004, p=0.5,
#                           Q=G_quarantine, beta_D=0.155, sigma_D=1/5.2,
#                           gamma_D=1/12.39, mu_D=0.0004,
#                           theta_E=0.02, theta_I=0.02, phi_E=0.2,
#                           phi_I=0.2, psi_E=1.0, psi_I=1.0, q=0.5,
#                           initI=10)
# checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal],
#                'p': [0.1, 0.5], 'theta_E': [0.02, 0.02],
#                'theta_I': [0.02, 0.02], 'phi_E': [0.2, 0.2],
#                'phi_I':   [0.2, 0.2]}
# model.run(T=300, checkpoints=checkpoints)
# model.figure_infections()
#}}}

# ------------------------------------------------------- PLOT
#{{{

ics = [S, E, I, R]
labels = ['S', 'E', 'I', 'R']
t = ts

plt.rcParams['savefig.facecolor'] = "0.8"
fig, ax = plt.subplots(2, 2, figsize=(20, 12))

#--- SIMU linear
for ic, lbl in zip(ics, labels):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[0,0], label=lbl)
    #sns.scatterplot(t, ic, ax=ax[0,0])

ax[0,0].set_xlabel('Time [days]', fontsize=22)
ax[0,0].set_ylabel('Number', fontsize=22)
ax[0,0].legend()
ax[0,0].grid()
ax[0,0].set_title('Simulation')
#---
ax[0,1].set(yscale="log")
ax[0,1].yaxis.set_major_formatter(\
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax[0,1].set_title('Simulation')

#--- SIMU log
for ic in ics:
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[0,1])
    #sns.scatterplot(t, ic, ax=ax[0,1])
ax[0,1].set_xlabel('Time [days]', fontsize=22)
ax[0,1].set_ylabel('Number', fontsize=22)
ax[0,1].grid()

#--- ANALYTIC linear
ics = [Sa, Ea, Ia, Ra]
labels = ['S', 'E', 'I', 'R']
t = Ta

for ic, lbl in zip(ics, labels):
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[1,0], label=lbl)
    #sns.scatterplot(t, ic, ax=ax[1,0])

ax[1,0].set_xlabel('Time [days]', fontsize=22)
ax[1,0].set_ylabel('Number', fontsize=22)
ax[1,0].legend()
ax[1,0].grid()
ax[1,0].set_title('Analytic')

#--- ANALYTIC log
ax[1,1].set(yscale="log")
ax[1,1].yaxis.set_major_formatter(\
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

for ic in ics:
    sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[1,1])
    #sns.scatterplot(t, ic, ax=ax[1,1])
ax[1,1].set_xlabel('Time [days]', fontsize=22)
ax[1,1].set_ylabel('Number', fontsize=22)
ax[1,1].legend()
ax[1,1].grid()
ax[1,1].set_title('Analytic')

#--- plt

plt.xticks(rotation=0, fontsize=22)
plt.yticks(rotation=90, fontsize=22)
plt.tight_layout()
fig.savefig('../plt/plot_sim.png')
plt.close()

#}}}
