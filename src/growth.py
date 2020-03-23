import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker



# based on
# https://www.wired.com/story/how-fast-does-a-virus-spread/

#########


def InfectionCurve(t, Ninit, loc, scale):

    N = Ninit
    ic = []
    t_old = t[0]

    for it in t:

      t_new = it
      dt = t_new - t_old

      r = np.random.normal(loc=loc, scale=scale, size=None)
      N = N + r * N * dt

      ic.append(N)

      t_old = t_new

      print([r, t_new, dt, N])

    return(ic)

def plt_IC(t, ic, fplot):

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.lineplot(x=t, y=ic, sort=False, linewidth=2)
    sns.scatterplot(t, ic)

    plt.suptitle("Infection curve", fontsize=16, fontweight='bold', color='white')
    plt.xticks(rotation=0)
    plt.xlabel('Time [days]')
    plt.ylabel('Number infected')

    fig.savefig(fplot)
    plt.close()
 

def plt_IC_n(t, ics, fplot):

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set(yscale="log")
    ax.yaxis.set_major_formatter(\
            ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    for ic in ics:
        sns.lineplot(x=t, y=ic, sort=False, linewidth=2)
        sns.scatterplot(t, ic)

    plt.suptitle("Infection curve", fontsize=16, fontweight='bold', color='white')
    plt.xticks(rotation=0)
    plt.xlabel('Time [days]')
    plt.ylabel('Number infected')

    fig.savefig(fplot)
    plt.close()
 


#################

fplot = '../plt/InfectionCurve.png'
          
# EXPERIMENTO 1
# Como afecta el error en el cálculo de R en la proyección de casos

start = 0.
stop = 40.
num = 41
dt = (stop-start)/num
t = np.linspace(start=start, stop=stop, num=num)

ics = []
for i in range(20):
    ic = InfectionCurve(t, 1., 0.2, 0.05)
    ics.append(ic)

fplot = '../plt/InfectionCurve_sd.png'
plt_IC_n(t, ics, fplot)


# EXPERIMENTO 2
# Como afecta R en el avance de los contagios

R = np.linspace(start=0.15, stop=0.3, num=10)
ics = []
for r in R:
    ic = InfectionCurve(t, 1., r, 0.0)
    ics.append(ic)

fplot = '../plt/InfectionCurve_R.png'
plt_IC_n(t, ics, fplot)
 



# keep track of:
#
# - numero real de contagiados
# - numero de casos confirmados
# - numero de casos recuerados
# - numero de fallecimientos
# - numero de pacientes leves (en la casa)
# - numero de pacientes moderados (internados, no UTI)
# - numero depacientes graves (UTI)


# infected
# confirmed
# recovered
# 
# inf_dead
# inf_uti
# inf_bed
# inf_home
# 



def InfectionCurve_full(t, Ninit, loc, scale):

    N = Ninit
    ic = []
    t_old = t[0]

    for it in t:

      t_new = it
      dt = t_new - t_old

      r = np.random.normal(loc=loc, scale=scale, size=None)
      N = N + r * N * dt

      ic.append(N)

      t_old = t_new

      print([r, t_new, dt, N])

    return(ic)
 
