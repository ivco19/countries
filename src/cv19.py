import numpy as np

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
 
def InfectionCurve_full(p):
    #{{{

    infected = []  # numero real de contagiados  
    confirmed = [] # numero de casos confirmados 
    recovered = [] # numero de casos recuerados  
    
    inf_dead = []  # numero de fallecimientos                           
    inf_home = []  # numero de pacientes leves (en la casa)             
    inf_bed = []   # numero de pacientes moderados (internados, no UTI) 
    inf_uti = []   # numero depacientes graves (UTI)                    
    
    ts = []
 
    n_infected = p.N_init

    fraction_report_sick = 0.8  # VERIFICAR

    t = 0.
    t_old = 0.

    while t < p.t_max:
        t_old = t
        t = t + p.dt
        ts.append(t)

        t_new = t

        #r = np.random.normal(loc=loc, scale=scale, size=None)
        n_infected = n_infected + p.R * n_infected * p.dt
        infected.append(N)

        i_delta_t_incubation = min(len(infected), p.t_incubation)
        N_become_sick = infected[-i_delta_t_incubation]
        n_confirmed = n_confirmed + N_become_sick * fraction_report_sick

        i_delta_t_incubation = min(len(infected), p.t_incubation)
        N_become_sick = infected[-i_delta_t_incubation]
        n_confirmed = n_confirmed + N_become_sick * fraction_report_sick

        

    return([ts, infected])
    #}}}

def plt_IC(t, ic, fplot):
    #{{{

    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

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
 
 

