import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


def check_file(sys_args):
    #{{{
    '''
    Parse paramenters for the simulation from a .ini file
    '''
    import sys
    from os.path import isfile

    if len(sys_args) == 2:    
        filename = sys_args[1]
        if isfile(filename):
            msg = "Loading configuration parameters from {}"
            print(msg.format(filename) )
        else:
            print("Input argument is not a valid file")
            print("Using default configuration file instead")
            filename = '../set/config.ini'
            #raise SystemExit(1) 
            
    else:
        print('Configuration file expected (just 1 argument)')
        print('example:  python run_correlation.py ../set/config.ini')
        print("Using default configuration file")
        #raise SystemExit(1) 
        filename = '../set/config.ini'

    return filename
    #}}}

def load_filenames(config):
    #{{{
    '''
    Parse paramenters for the simulation from a .ini file
    '''

    from collections import namedtuple
    import numpy as np

    # Experiment settings
    #-----------------------
    
    exp_ID = config['experiment']['exp_id']

    dir_plot = config['experiment']['dir_plot'] 

    ext = config['experiment']['extension'] 

    fname_infected =  config['experiment']['fname_infected'] 
    fname_confirmed = config['experiment']['fname_confirmed']
    fname_recovered = config['experiment']['fname_recovered']
    fname_inf_dead =  config['experiment']['fname_inf_dead'] 
    fname_inf_home =  config['experiment']['fname_inf_home'] 
    fname_inf_bed =   config['experiment']['fname_inf_bed']
    fname_inf_uti =   config['experiment']['fname_inf_uti']
     
    fname_infected =  dir_plot + fname_infected + '_' + exp_ID + ext
    fname_confirmed = dir_plot + fname_confirmed + '_' + exp_ID + ext
    fname_recovered = dir_plot + fname_recovered + '_' + exp_ID + ext
    fname_inf_dead =  dir_plot + fname_inf_dead + '_' + exp_ID + ext 
    fname_inf_home =  dir_plot + fname_inf_home + '_' + exp_ID + ext  
    fname_inf_bed =   dir_plot + fname_inf_bed + '_' + exp_ID + ext 
    fname_inf_uti =   dir_plot + fname_inf_uti + '_' + exp_ID + ext 

    names = 'fname_infected \
             fname_confirmed \
             fname_recovered \
             fname_inf_dead \
             fname_inf_home \
             fname_inf_bed \
             fname_inf_uti'
    
    parset = namedtuple('pars', names)

    res = parset(fname_infected, fname_confirmed, fname_recovered,
                 fname_inf_dead, fname_inf_home, fname_inf_bed, fname_inf_uti) 

    return(res)  

    #}}}                

def load_parameters(config):
    #{{{
    '''
    Parse paramenters for the simulation from a .ini file
    '''

    from collections import namedtuple
    import numpy as np

    # Experiment settings
    #-----------------------
    
    dir_data = config['experiment']['dir_data']
    dir_plot = config['experiment']['dir_plot']
    
    t_max = float(config['experiment']['t_max'])
    dt = float(config['experiment']['dt'])


    # Transmision dynamics
    #-------------------
    
    # population
    population = int(config['transmision']['population'])
    N_init = int(config['transmision']['n_init'])
    R = float(config['transmision']['r'])

    intervention_start = config['transmision']['intervention_start']
    intervention_end = config['transmision']['intervention_end']
    intervention_decrease = config['transmision']['intervention_decrease']
    intervention_start =     float(intervention_start)
    intervention_end =       float(intervention_end)
    intervention_decrease =  float(intervention_decrease)
    
    t_incubation = float(config['transmision']['t_incubation'])
    t_infectious = float(config['transmision']['t_infectious'])
    
    # Clinical dynamics
    #-------------------
    
    #---# Morbidity statistics
    
    morbidity_file = config['clinical']['morbidity_file']
    
    t_death = config['clinical']['t_death']
    bed_stay = config['clinical']['bed_stay']
    mild_recovery = config['clinical']['mild_recovery']
    bed_rate = config['clinical']['bed_rate']
    bed_wait = config['clinical']['bed_wait']
    

    names = 'dir_data dir_plot t_max dt population \
     N_init R \
     intervention_start intervention_end intervention_decrease \
     t_incubation t_infectious'
    
    parset = namedtuple('pars', names)

    res = parset(dir_data, dir_plot, t_max, dt, population, \
          N_init, R, \
          intervention_start, intervention_end, intervention_decrease, \
          t_incubation, t_infectious)

    return(res)

    #}}}                
 
def InfectionCurve_full_0(p):
    #{{{

    population = 100

    # daily changes
    n_I = p.N_init
    n_S = n_C = n_R = n_H = n_B = n_U = n_D = 0

    # cumulative time series
    I = [n_I] # Infected
    S = [n_S] # Sick or present synthoms
    C = [n_C] # Confirmed
    R = [n_R] # Recovered
    H = [n_H] # sick at Home
    B = [n_B] # sick in Bed in Health System
    U = [n_U] # sick in Intensive Care Unit
    D = [n_D] # dead by virus

    ts = [0.] # time series
 
    # delay times (number in units of days!)
    # delay times are set in number of time steps
    T_IC = int(5. /p.dt)   # incubation time (from infection to synthoms)

    T_HB = int(10./p.dt)   # time from synthoms to hospitalization (Bed)
    T_HU = int(10./p.dt)   # time from synthoms to hospitalization (ICU)
    T_HR = int(11./p.dt)   # Recovery time for mild cases
    T_HD = int(10./p.dt)   # Time length for death at home

    T_BH = int(10./p.dt)   # Stay in hospital until give away
    T_BU = int(10./p.dt)   # Stay in hospital until transference to ICU
    T_BR = int(15./p.dt)   # Length of hospital (ICU) stay              
    T_BD = int(20./p.dt)   # Time in hospital until dead (without ICU)

    T_UB = int(10./p.dt)   # Time in ICU until transference to bed
    T_UH = int(28./p.dt)   # stay in ICU stay until give away
    T_UR = int(28./p.dt)   # Length of hospital (ICU) stay until recovered
    T_UD = int(28./p.dt)   # Stay in ICU until death


    # fractions for transitions
    f_IC = 0.95   # probability an infected person gets sick
                
    f_HB = 0.1    # probability of hospitalization
    f_HU = 0.     # probability of emergency hospitalization
    f_HR = 0.9    # probability of recovery at home
    f_HD = 1. - f_HB - f_HU - f_HR  # probability of death in home
               
    f_BH = 0.0    # probability of give away before recovery
    f_BU = 0.2    # probability of transference to ICU
    f_BR = 0.8    # probability of recovery in hospital
    f_BD = 1. - f_BH - f_BU - f_BR  # probability of death in common bed

    f_UB = 0.6     # probability of transference from ICU to common bed
    f_UH = 0.0     # probability of give away from ICU
    f_UR = 0.0     # probability of recovery from ICU
    f_UD = 1. - f_BH - f_BU - f_BR # probability of death in ICU

    t = 0.
    time_steps = 0

    while t < p.t_max:

        time_steps = time_steps + 1

        t_prev = t
        t = t + p.dt
        ts.append(t)
 

        if time_steps > 1:
            n_I = I[-1] * p.R * p.dt


        print(T_IC, time_steps, len(I))

        i_IC = I[-T_IC] if T_IC < len(I) else 0
        #i_

        #n_C = i_IC * f_IC - 

        #I.append(n_I)
        #C.append(n_C)

        #n_S = 0
        #n_R = 0
        #n_H = 0
        #n_B = 0
        #n_U = 0
        #n_D = 0 


    return([ts, I, C])
    #}}}

def InfectionCurve_full(p):
    #{{{
    from graph_tools import Graph

    g = Graph()

    for node in ['I','C','R','H','B','U','D',]:
        g.add_node(node, 0)

    g.set_node('I', p.N_init)
    
    # cumulative time series
    I = [g.get_node_value('I')] # Infected
    C = [g.get_node_value('C')] # Confirmed                    
    R = [g.get_node_value('R')] # Recovered                    
    H = [g.get_node_value('H')] # sick at Home                 
    B = [g.get_node_value('B')] # sick in Bed in Health System 
    U = [g.get_node_value('U')] # sick in Intensive Care Unit  
    D = [g.get_node_value('D')] # dead by virus                

    ts = [0.] # time series
    nms = ['prob','lag']
    p_dt = 1.
 
    # delay times (number in units of days!)
    # delay times are set in number of time steps
    T_IC = int(5. /p_dt)   # incubation time (from infection to synthoms)

    T_CH = 0
    T_CB = 0
    T_CU = 0
    T_CD = 0

    T_HB = int(10./p_dt)   # time from synthoms to hospitalization (Bed)
    T_HU = int(10./p_dt)   # time from synthoms to hospitalization (ICU)
    T_HR = int(11./p_dt)   # Recovery time for mild cases
    T_HD = int(10./p_dt)   # Time length for death at home

    T_BH = int(10./p_dt)   # Stay in hospital until give away
    T_BU = int(10./p_dt)   # Stay in hospital until transference to ICU
    T_BR = int(15./p_dt)   # Length of hospital (ICU) stay              
    T_BD = int(20./p_dt)   # Time in hospital until dead (without ICU)

    T_UB = int(10./p_dt)   # Time in ICU until transference to bed
    T_UH = int(28./p_dt)   # stay in ICU stay until give away
    T_UR = int(28./p_dt)   # Length of hospital (ICU) stay until recovered
    T_UD = int(28./p_dt)   # Stay in ICU until death

    # fractions for transitions
    f_IC = 0.95   # probability an infected person gets sick

    f_CH = 0.95   # probability of a domiciliary confirmation
    f_CB = 0.05   # probability of a confirmation in health system
    f_CU = 0.00   # probability of a confirmation in ICU
    f_CD = 1. - f_CH - f_CB - f_CU   # probability of a confirmation in autopsy
                
    f_HB = 0.1    # probability of hospitalization
    f_HU = 0.     # probability of emergency hospitalization
    f_HR = 0.9    # probability of recovery at home
    f_HD = 1. - f_HB - f_HU - f_HR  # probability of death in home
               
    f_BH = 0.0    # probability of give away before recovery
    f_BU = 0.2    # probability of transference to ICU
    f_BR = 0.8    # probability of recovery in hospital
    f_BD = 1. - f_BH - f_BU - f_BR  # probability of death in common bed

    f_UB = 0.6     # probability of transference from ICU to common bed
    f_UH = 0.0     # probability of give away from ICU
    f_UR = 0.0     # probability of recovery from ICU
    f_UD = 1. - f_BH - f_BU - f_BR # probability of death in ICU
                                                                    

    g.add_edge('I', 'I', nms, [p.R,  0])
    
    g.add_edge('I', 'C', nms, [f_IC, T_IC])

    g.add_edge('C', 'H', nms, [f_CH, T_CH])
    g.add_edge('C', 'B', nms, [f_CB, T_CB])
    g.add_edge('C', 'U', nms, [f_CU, T_CU])

    g.add_edge('H', 'B', nms, [f_HB, T_HB])
    g.add_edge('H', 'U', nms, [f_HU, T_HU])
    g.add_edge('H', 'R', nms, [f_HR, T_HR])
    g.add_edge('H', 'D', nms, [f_HD, T_HD])

    g.add_edge('B', 'H', nms, [f_BH, T_BH])
    g.add_edge('B', 'U', nms, [f_BU, T_BU])
    g.add_edge('B', 'R', nms, [f_BR, T_BR])
    g.add_edge('B', 'D', nms, [f_BD, T_BD])

    g.add_edge('U', 'B', nms, [f_UB, T_UB])
    g.add_edge('U', 'H', nms, [f_UH, T_UH])
    g.add_edge('U', 'R', nms, [f_UR, T_UR])
    g.add_edge('U', 'D', nms, [f_UD, T_UD])

    t = 0.
    time_steps = 0

    while t < p.t_max:

        time_steps = time_steps + 1

        t_prev = t
        t = t + p.dt
        ts.append(t)


        # activation of all nodes
        v = g.get_node('I')
        prob = g.node_activation('I', 'prob')[0]
        lag = g.node_activation('I', 'lag')[0]

        ilag = -lag if lag < len(I) else 1
        n_I = I[-1] + I[ilag] * prob * p.dt
        I.append(n_I)

        #### TO DO:
        # completar las demas variables (nodos)
        # ver si se puede escribir por comprension


    return([ts, I])
    #}}}

def plt_IC(t, ic, fplot):
    #{{{

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set(yscale="log")
    ax.yaxis.set_major_formatter(\
            ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    sns.lineplot(x=t, y=ic, sort=False, linewidth=2)
    sns.scatterplot(t, ic)

    plt.suptitle("Infection curve", fontsize=16, fontweight='bold', color='white')
    plt.xticks(rotation=0)
    plt.xlabel('Time [days]')
    plt.ylabel('Number infected')

    fig.savefig(fplot)
    plt.close()
    #}}}
                        
def plt_IC_n(t, ics, fplot):
    #{{{

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
    #}}}
 
 


# TRY graphs

# https://www.python-course.eu/graphs_python.php
# https://www.bogotobogo.com/python/python_graph_data_structures.php
# https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
