import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from configparser import ConfigParser


class parser(ConfigParser):
    #{{{
    """
    parser class
    manipulation of parser from ini files
    """
    def check_file(self, sys_args):
        '''
        chek_file(args): 
        Parse paramenters for the simulation from a .ini file
        
        Args:
            filename (str): the file name of the map to be read

        Raises:

        Returns:
            readmap: a healpix map, class ?
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
    
        self.filename = filename
    
    def read_config_file(self):
        '''
        chek_file(args): 
        Parse paramenters for the simulation from a .ini file
        
        Args:
            filename (str): the file name of the map to be read

        Raises:

        Returns:
            readmap: a healpix map, class ?
        ''' 

        import sys
        from os.path import isfile
    
        self.read(self.filename)

    def load_filenames(self):
        '''
        load_filenames(self): 
        make filenames based on info in config file
        
        Args:
            None

        Raises:

        Returns:
            list of filenames
        ''' 
    
        from collections import namedtuple
        import numpy as np
    
        # Experiment settings
        #-----------------------
        
        exp_ID = self['experiment']['exp_id']
    
        dir_plot = self['experiment']['dir_plot'] 
    
        ext = self['experiment']['extension'] 
    
        fname_infected =  self['experiment']['fname_infected'] 
        fname_confirmed = self['experiment']['fname_confirmed']
        fname_recovered = self['experiment']['fname_recovered']
        fname_inf_dead =  self['experiment']['fname_inf_dead'] 
        fname_inf_home =  self['experiment']['fname_inf_home'] 
        fname_inf_bed  =  self['experiment']['fname_inf_bed']
        fname_inf_uti  =  self['experiment']['fname_inf_uti']
        fname_sir      =  self['experiment']['fname_sir']
        fname_seir     =  self['experiment']['fname_seir']
         
        fname_infected =  dir_plot + fname_infected + '_' + exp_ID + ext
        fname_confirmed = dir_plot + fname_confirmed + '_' + exp_ID + ext
        fname_recovered = dir_plot + fname_recovered + '_' + exp_ID + ext
        fname_inf_dead =  dir_plot + fname_inf_dead + '_' + exp_ID + ext 
        fname_inf_home =  dir_plot + fname_inf_home + '_' + exp_ID + ext  
        fname_inf_bed =   dir_plot + fname_inf_bed + '_' + exp_ID + ext 
        fname_inf_uti =   dir_plot + fname_inf_uti + '_' + exp_ID + ext 
        fname_sir = dir_plot + fname_sir + '_' + exp_ID + ext 
        fname_seir = dir_plot + fname_seir + '_' + exp_ID + ext 
    
        names = 'fname_infected \
                 fname_confirmed \
                 fname_recovered \
                 fname_inf_dead \
                 fname_inf_home \
                 fname_inf_bed \
                 fname_inf_uti \
                 fname_sir \
                 fname_seir'
        
        parset = namedtuple('pars', names)
    
        res = parset(fname_infected, fname_confirmed, fname_recovered,
                     fname_inf_dead, fname_inf_home, fname_inf_bed, fname_inf_uti,
                     fname_sir, fname_seir) 
    
        self.filenames = res
    
    def load_parameters(self):
        '''
        load_parameters(self): 
        load parameters from config file
        
        Args:
            None

        Raises:

        Returns:
            list of parameters as a named tuple
        
        ''' 
        
        from collections import namedtuple
        import numpy as np
    
        # Experiment settings
        #-----------------------
        
        dir_data = self['experiment']['dir_data']
        dir_plot = self['experiment']['dir_plot']
        
        t_max = float(self['experiment']['t_max'])
        dt = float(self['experiment']['dt'])
    
        # Transmision dynamics
        #---------------------
        
        # population
        population = int(self['transmision']['population'])
        N_init = int(self['transmision']['n_init'])
        R = float(self['transmision']['r'])
    
        intervention_start = self['transmision']['intervention_start']
        intervention_end = self['transmision']['intervention_end']
        intervention_decrease = self['transmision']['intervention_decrease']
        intervention_start =     float(intervention_start)
        intervention_end =       float(intervention_end)
        intervention_decrease =  float(intervention_decrease)
        
        t_incubation = float(self['transmision']['t_incubation'])
        t_infectious = float(self['transmision']['t_infectious'])
        
        # Clinical dynamics
        #-------------------
        
        #---# Morbidity statistics
        
        morbidity_file = self['clinical']['morbidity_file']
        
        t_death = self['clinical']['t_death']
        bed_stay = self['clinical']['bed_stay']
        mild_recovery = self['clinical']['mild_recovery']
        bed_rate = self['clinical']['bed_rate']
        bed_wait = self['clinical']['bed_wait']
        
    
        names = 'dir_data dir_plot t_max dt population \
         N_init R \
         intervention_start intervention_end intervention_decrease \
         t_incubation t_infectious'
        
        parset = namedtuple('pars', names)
    
        res = parset(dir_data, dir_plot, t_max, dt, population, \
              N_init, R, \
              intervention_start, intervention_end, intervention_decrease, \
              t_incubation, t_infectious)
    
        self.p = res
    #}}}


class table_draw:
    #{{{

    def __init__(self):
        self.size = []
        self.ranges = []
        self.prob = []
        self.dif_prob = []

    def load(self, filename):
        D = pd.read_csv(filename)
        self.D = D
        self.size = D.shape[0]

    def add_clean_column(self, columnname, column):
        self.D[columnname] = column

    def tabular(self, x, y):
        self.x = self.d[x]
        self.p = self.d[y]
        
    def random_gen(self, xinfname, xsupname, yname):
        """
        method: random_gen
           Generates a random sample from a given probability density
           The probability density must be in the table, in the 
           table_draw class.
        """

        import sys
        u = random()
        j = self.size

        #msg = 'testing if %f is in interval (%f %f) / index: %i'
        y_old = 0.
        for i, y_new in enumerate(self.D[yname]):
            if u > y_old and u < y_new:
                j = i
                #print(msg % (u, y_old, y_new, i))
                break
            else:
                y_old = y_new

        if j<0 or j>= self.size: 
            print(j, self.size)
            sys.exit()

        x1 = self.D[xinfname][j]
        x2 = self.D[xsupname][j]
        res = (u-y_old)/(y_new-y_old)*(x2-x1) + x1
        res = int(res)
        return(res)

    def test_random_gen(self, nran, fplot):

        r = []
        for _ in range(nran):
            r.append(self.random_gen('x_inf', 'x_sup','y'))

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
       
        plt.hist(r)
        plt.suptitle("random gen test", fontsize=16, \
                fontweight='bold', color='white')
        plt.xticks(rotation=0)
        plt.xlabel('X')
        plt.ylabel('PDF')
       
        fig.savefig(fplot)
        plt.close()
       
    def save_clean_CDF(self, filename):

        df = self.D[['x_inf', 'x_sup','y']]
        df.to_csv(filename)
    #}}}


class node:
    #{{{
    """
    class node
    This class is used to create and manipulated nodes.

    """
    def __init__(self, nnode, value):
        self.id = nnode
        self.value = value
        self.outgoing = {}
        self.incoming = {}

    def __str__(self):
        # (para que funcione el print)
        string = str(self.id) + ' node, outgoing: ' + \
                 str([x.id for x in self.outgoing]) + \
                 ' incoming: ' + str([x.id for x in self.incoming])
        return string

    def add_neighbor(self, neighbor, names, values):
        self.outgoing[neighbor] = {}
        for n, v in zip(names, values):
            self.outgoing[neighbor][n] = v
        #print(neighbor.id, '---->', self)
        #print(self.id)  # (for testing purposes)
 
    def be_neighbor(self, neighbor, names, values):
        self.incoming[neighbor] = {}
        for n, v in zip(names, values):
            self.incoming[neighbor][n] = v
        #print(neighbor.id, '- - >', self)

    def get_connections(self):
        return self.outgoing.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.outgoing[neighbor]
    #}}}


class Graph:
    #{{{
    """
    class Graph
    This class is used to create and manipulated graphs
    It makes a heavy use of the node class 
    A graph is made of nodes and edges.  This class allows to store
    a value for each node and different "weights" for each edge.
    Also, edges are directed.  

    Exampe of usage:
    g = Graph()

    for i, inode in enumerate(['A','B','C','D']):
        print(i)
        g.add_node(inode, 0)

    nms = ['x', 'y']
    g.add_edge('A', 'B', nms, [1, 100])
    g.add_edge('A', 'C', nms, [2, 200])
    g.add_edge('B', 'D', nms, [3, 300])
    g.add_edge('D', 'B', nms, [4, 400])
    g.add_edge('D', 'C', nms, [5, 500])
    g.add_edge('C', 'C', nms, [6, 600])

    # A node can be connected to itself.
    g.add_edge('B', 'B', nms, [333, 333])

    g.show()


    Attributes
    ----------
    vert_dict: dict
        a dict containing the vertices
    num_vertices : int
        the number of nodes (or vertices) in a graph

    Methods
    -------
    add_node(node, value)
    get_node()
    get_node_value()
    set_node()
    get_node_ids()
    get_nodes_to()
    get_nodes_from()


    """
    def __init__(self):
        """
        init a Graph object
        """
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    # node functions --------------------------------------
    def add_node(self, nnode, value):
        """
        method: add_node
     
        Adds a node to a graph. The node must have a value.

        """
        self.num_vertices = self.num_vertices + 1
        new_node = node(nnode, value)
        self.vert_dict[nnode] = new_node
        return new_node

    def get_node(self, n):
        """
        method: get_node
     
        Parameters
        ----------
           n: str
        Returns
        -------
           node: a node object
        """

        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def get_node_value(self, n):
        """
        method: get_node_value
     
        Parameters
        ----------
           n: str

        Returns
        -------
           value: float
        """
 
        if n in self.vert_dict:
            return self.vert_dict[n].value
        else:
            return None

    def set_node(self, n, value):
        """
        method: set_node
     
        Parameters
        ----------
           n: str
              The ID or name of the node
           value: float
              The value to be assigned to the node

        Returns
        -------
           updates the Graph
        """
   
        if n in self.vert_dict:
            v = self.get_node(n)
            v.value = value
        else:
            return None

    def get_node_ids(self):
        keys = self.vert_dict.keys()
        l = list(keys)
        return(l)

    def get_nodes_to(self, nnode):
        v = self.get_node(nnode)
        c = []
        for i in v.incoming:
            c.append(i.id)
        return(c)

    def get_nodes_from(self, nnode):
        v = self.get_node(nnode)
        c = []
        for i in v.outgoing:
            c.append(i.id)
        return(c)


    # edge functions --------------------------------------
    def add_edge(self, frm, to, names = [], values = 0):
        """
        warning: does not verify if edge already exists
        """
        mis1 = frm not in self.vert_dict
        mis2 = to not in self.vert_dict
        if mis1 or mis2:
            print('Not a node called ')
            print(frm)
            print(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], names, values)

        self.vert_dict[to].be_neighbor(self.vert_dict[frm], names, values)

    def get_edge(self, frm, to, field):
        if frm not in self.vert_dict:
            self.add_node(frm)
        if to not in self.vert_dict:
            self.add_node(to)

        v_frm = self.get_node(frm)
        v_to  = self.get_node(to)

        ws = v_frm.get_weight(v_to)
        value = ws[field]
        return(value)

 
    # graph functions --------------------------------------
    def show_weights(self):
        for v in self:
            print('NODE %s:  %f' % (v.id, v.value))
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                print ('             %s -> %s %s' % (vid, wid, v.get_weight(w)))

    def show_connections(self):
        node_list = self.get_node_ids()
        for inode in node_list:
            print('nodes from %s: ' % inode, end='')
            print(self.get_nodes_from(inode))
            print('\nnodes to %s: ' % inode, end='')
            print(self.get_nodes_to(inode))

    # computation functions --------------------------------
    def node_activation(self, nnode, key):
        l = self.get_nodes_to(nnode)
        x = []
        for v in l:
            a = self.get_node(v)
            x.append( self.get_edge(a.id, nnode, key) )
        return(x)
        # idea:
        # activation of all nodes
        # v = g.get_node('I')
        #prob = g.node_activation('I', 'prob')[0]
        #lag = g.node_activation('I', 'lag')[0]

    def node_upgrade(self, nnode, key):
        l = self.get_nodes_to(nnode)
        x = []
        for v in l:
            a = self.get_node(v)
            x.append( self.get_edge(a.id, nnode, key) )
        return(x)
    #}}}


class InfectionCurve:

    # see:
    # http://epirecip.es/epicookbook/chapters/sir-stochastic-discretestate-discretetime/python
        
    def model_SIR(self, p):
        #{{{
        """
        method: model_SIR(parameters)

        This function implements a SIR model without vital dynamics
        under the assumption of a closed population.
        Recovered individuals become immune for ever.
        In this model exposed individuals become instantly infectious,
        i.e., there is no latency period like in the SEIR model.
     
        Parameters
        ----------
           p: parameters containing transition probabilities
              population number

        Returns
        -------
           value: Time series for S, I and R
        """
        g = Graph()
    
        for node in ['I','C','R','H','B','U','D',]:
            g.add_node(node, 0)
    
        g.set_node('I', p.N_init)
        
        # cumulative time series
        I = [g.get_node_value('I')] # Infected
        C = [g.get_node_value('C')] # Confirmed                    
        R = [g.get_node_value('R')] # Recovered                    
    
        ts = [0.] # time series
        nms = ['prob','lag']
        p_dt = 1.
     
        
        # En este modelo todos los infectados se confirman a los 10
        # dias y se curan a los 20 dias de confirmados
        T_IC = int(p.t_incubation/ p.dt)
        T_CR = 20
        f_IC = 1.
        f_CR = 1.
    
        g.add_edge('I', 'I', nms, [p.R,  0])
        g.add_edge('I', 'C', nms, [f_IC, T_IC])
        g.add_edge('C', 'R', nms, [f_CR, T_CR])
    
        t = 0.
        time_steps = 0
    
        while t < p.t_max:
    
            time_steps = time_steps + 1
    
            t_prev = t
            t = t + p.dt
            ts.append(t)
    
            # (( I ))
            prob_II = g.get_edge('I', 'I', 'prob')
            lag_II = g.get_edge('I', 'I', 'lag')
            update_II = I[-lag_II] if lag_II < len(I) else 0.

            prob_IC = g.get_edge('I', 'C', 'prob')
            lag_IC = g.get_edge('I', 'C', 'lag')
            update_IC = I[-lag_IC] if lag_IC < len(I) else 0.

            n_I = min(I[-1] + I[-1] * prob_II * p.dt, p.population) - \
                  update_IC * prob_IC * p.dt 
            n_I = max(n_I, 0)

            I.append(n_I)

            # (( C ))
            prob_CR = g.get_edge('C', 'R', 'prob')
            lag_CR = g.get_edge('C', 'R', 'lag')
            update_CR = C[-lag_CR] if lag_CR < len(C) else 0.

            n_C = min(C[-1] + update_IC * prob_IC * p.dt, p.population) - \
                  update_CR * prob_CR * p.dt
            n_C = max(n_C, 0)
            C.append(n_C)

            # (( R ))
            n_R = min(R[-1] + update_CR * prob_CR * p.dt, p.population) # recuperados nuevos
            n_R = max(n_R, 0)
            R.append(n_R)
 
        return([ts, I, C, R])
        #}}}

    def model_SEIR(self, p):
        #{{{
        """
        method: model_SEIR(parameters)

        This function implements a SEIR model without vital dynamics
        under the assumption of a closed population.
        Recovered individuals become immune for ever.
        ref.: https://www.idmod.org/docs/hiv/model-seir.html
 
        Parameters
        ----------
           p: parameters containing transition probabilities
              population number

        Returns
        -------
           value: Time series for S, E, I and R 
        """
        g = Graph()
    
        for node in ['S','E','I','R']:
            g.add_node(node, 0)
    
        g.set_node('S', p.population)
        g.set_node('E', 0)
        g.set_node('I', p.N_init)
        g.set_node('R', 0)
        
        # cumulative time series
        S = [g.get_node_value('S')] # Susceptible
        E = [g.get_node_value('E')] # Exposed
        I = [g.get_node_value('I')] # Infected
        R = [g.get_node_value('R')] # Recovered
        
        ts = [0.] # time series
        nms = ['prob','lag']
        p_dt = 1.
     
        T_IC = int(p.t_incubation/ p.dt)
    
        g.add_edge('S', 'S', nms, [0.1,  2])
        g.add_edge('E', 'E', nms, [0.4,  21])
        g.add_edge('I', 'I', nms, [0.1,  2])

        g.add_edge('S', 'E', nms, [1.2,  1])
        g.add_edge('E', 'I', nms, [0.1,  14]) #[, tiempo de incubacion]
        g.add_edge('I', 'R', nms, [0.7, 2]) #[, tiempo de recuperacion] 
    
        t = 0.
        time_steps = 0
    
        while t < p.t_max:
    
            time_steps = time_steps + 1
    
            t_prev = t
            t = t + p.dt
            ts.append(t)
    
            # (( S ))
            prob_SS = g.get_edge('S', 'S', 'prob') # beta
            lag_SS = g.get_edge('S', 'S', 'lag')
            update_SS = S[-lag_SS] if lag_SS < len(S) else 0.

            dS = - S[-1] * ( I[-1]/p.population ) * prob_SS 
            #n_S = min(S[-1] + min(dS*p.dt, 0), p.population)
            n_S = S[-1] + dS*p.dt

            # (( E ))
            prob_EE = g.get_edge('E', 'E', 'prob')
            lag_EE = g.get_edge('E', 'E', 'lag')
            update_EE = E[-lag_EE] if lag_EE < len(E) else 0.

            dE = - dS - prob_EE * E[-1]

            #n_E = min(E[-1] + max(dE*p.dt, 0), p.population)
            n_E = E[-1] + dE*p.dt

            # (( I ))
            prob_EI = g.get_edge('E', 'I', 'prob')
            lag_EI = g.get_edge('E', 'I', 'lag')
            update_EI = E[-lag_EI] if lag_EI < len(E) else 0.

            prob_IR = g.get_edge('I', 'R', 'prob')
            lag_IR = g.get_edge('I', 'R', 'lag')
            update_IR = I[-lag_IR] if lag_IR < len(I) else 0.

            prob_II = g.get_edge('I', 'I', 'prob')

            dI = prob_EI * update_EI - prob_IR * update_IR
            dI = -dI  # porque ????
            n_I = min(I[-1] + dI*p.dt, p.population)


            # (( R ))
            prob_II = g.get_edge('I', 'I', 'prob')
            dR = prob_II * I[-1]
            n_R = min(R[-1] + max(dR*p.dt, 0), p.population)

            S.append(n_S)
            E.append(n_E)
            I.append(n_I)
            R.append(n_R)

        return([ts, S, E, I, R]) 
        #}}}

    def model_SEIRF(self, p):
        #{{{
        g = Graph()
    
        for node in ['S','E','I','R','F']:
            g.add_node(node, 0)
    
        g.set_node('S', p.population)
        g.set_node('E', 0)
        g.set_node('I', p.N_init)
        g.set_node('R', 0)
        g.set_node('F', 0)
        
        # cumulative time series
        S = [g.get_node_value('S')] # Susceptible
        E = [g.get_node_value('E')] # Exposed
        I = [g.get_node_value('I')] # Infected
        R = [g.get_node_value('R')] # Recovered
        F = [g.get_node_value('F')] # Fatalities
        
        ts = [0.] # time series
        nms = ['prob','lag']
        p_dt = 1.
     
        
        T_IC = int(p.t_incubation/ p.dt)
    
        g.add_edge('S', 'E', nms, [p.R,  0])
        g.add_edge('E', 'E', nms, [p.R,  0])
        g.add_edge('E', 'I', nms, [0.5,  10])
        g.add_edge('I', 'I', nms, [0.5,  10])
        g.add_edge('I', 'R', nms, [0.98, 30])
        g.add_edge('I', 'S', nms, [0.98, 30])
        g.add_edge('R', 'F', nms, [0.02, 30])
    
        t = 0.
        time_steps = 0
    
        while t < p.t_max:
    
            time_steps = time_steps + 1
    
            t_prev = t
            t = t + p.dt
            ts.append(t)
             
            # (( S ))
            prob_IS = g.get_edge('I', 'S', 'prob') # beta
            dS = - S[-1] * ( I[-1]/p.population ) * prob_IS 
            n_S = S[-1] + dS*p.dt

            # (( E ))
            prob_EE = g.get_edge('E', 'E', 'prob')
            lag_EE = g.get_edge('E', 'E', 'lag')
            update_EE = E[-lag_EE] if lag_EE < len(E) else 0.

            dE = - dS - prob_EE * update_EE
            n_E = E[-1] + dE*p.dt

            # (( I ))
            prob_EI = g.get_edge('E', 'I', 'prob')
            lag_EI = g.get_edge('E', 'I', 'lag')
            update_EI = E[-lag_EI] if lag_EI < len(E) else 0.

            prob_II = g.get_edge('I', 'I', 'prob')
            lag_II = g.get_edge('I', 'I', 'lag')
            update_II = I[-lag_II] if lag_II < len(I) else 0.

            prob_IR = g.get_edge('I', 'R', 'prob')
            lag_IR = g.get_edge('I', 'R', 'lag')
            update_IR = I[-lag_IR] if lag_IR < len(I) else 0.

            prob_II = g.get_edge('I', 'I', 'prob')

            dI = prob_EI * update_EI + prob_II * update_II - prob_IR * update_IR
            n_I = min(I[-1] + dI*p.dt, p.population)

            # (( R ))
            dR = prob_IR * update_IR
            n_R = min(R[-1] + max(dR*p.dt, 0), p.population)

            # (( F ))
            prob_RF = g.get_edge('R', 'F', 'prob')
            lag_RF = g.get_edge('R', 'F', 'lag')
            update_RF = I[-lag_RF] if lag_RF < len(R) else 0.
            
            dF = prob_RF * R[-1]
            n_F = min(R[-1] + max(dR*p.dt, 0), p.population)

            S.append(n_S)
            E.append(n_E)
            I.append(n_I)
            R.append(n_R)
            F.append(n_F)
 


        return([ts, S, E, I, R, F]) 
        #}}}

    def model_SIER_BH(self, p):
        #{{{ 
        '''
        InfectionCurve(self, p): 
        computes the Infection Curve based on a probabilistic model
        implemented in a simulation
        
        Args:
            config object from ParseConfig
      
        Raises:
      
        Returns:
            Time series for the curves of:
            - Infected
            - Confirmed
            - Recovered
            - Confirmed at home
            - Confirmed at hospital
            - Confirmed at ICU
            - Dead
        
        ''' 

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
        T_CR = 0
    
        #T_HB = int(10./p_dt)   # time from synthoms to hospitalization (Bed)
        #T_HU = int(10./p_dt)   # time from synthoms to hospitalization (ICU)
        #T_HR = int(11./p_dt)   # Recovery time for mild cases
        #T_HD = int(10./p_dt)   # Time length for death at home
    
        #T_BH = int(10./p_dt)   # Stay in hospital until give away
        #T_BU = int(10./p_dt)   # Stay in hospital until transference to ICU
        #T_BR = int(15./p_dt)   # Length of hospital (ICU) stay              
        #T_BD = int(20./p_dt)   # Time in hospital until dead (without ICU)
    
        #T_UB = int(10./p_dt)   # Time in ICU until transference to bed
        #T_UH = int(28./p_dt)   # stay in ICU stay until give away
        #T_UR = int(28./p_dt)   # Length of hospital (ICU) stay until recovered
        #T_UD = int(28./p_dt)   # Stay in ICU until death
    
        ## fractions for transitions
        #f_IC = 0.95   # probability an infected person gets sick
    
        #f_CH = 0.95   # probability of a domiciliary confirmation
        #f_CB = 0.05   # probability of a confirmation in health system
        #f_CU = 0.00   # probability of a confirmation in ICU
        #f_CD = 1. - f_CH - f_CB - f_CU   # probability of a confirmation in autopsy
        #f_CR = 1.
        #            
        #f_HB = 0.1    # probability of hospitalization
        #f_HU = 0.     # probability of emergency hospitalization
        #f_HR = 0.9    # probability of recovery at home
        #f_HD = 1. - f_HB - f_HU - f_HR  # probability of death in home
        #           
        #f_BH = 0.0    # probability of give away before recovery
        #f_BU = 0.2    # probability of transference to ICU
        #f_BR = 0.8    # probability of recovery in hospital
        #f_BD = 1. - f_BH - f_BU - f_BR  # probability of death in common bed
    
        #f_UB = 0.6     # probability of transference from ICU to common bed
        #f_UH = 0.0     # probability of give away from ICU
        #f_UR = 0.0     # probability of recovery from ICU
        #f_UD = 1. - f_BH - f_BU - f_BR # probability of death in ICU
        #                                                                
    
        #g.add_edge('I', 'I', nms, [p.R,  0])
        #
        #g.add_edge('I', 'C', nms, [f_IC, T_IC])
    
        #g.add_edge('C', 'H', nms, [f_CH, T_CH])
        #g.add_edge('C', 'B', nms, [f_CB, T_CB])
        #g.add_edge('C', 'U', nms, [f_CU, T_CU])
        #g.add_edge('C', 'R', nms, [f_CR, T_CR])
    
        #g.add_edge('H', 'B', nms, [f_HB, T_HB])
        #g.add_edge('H', 'U', nms, [f_HU, T_HU])
        #g.add_edge('H', 'R', nms, [f_HR, T_HR])
        #g.add_edge('H', 'D', nms, [f_HD, T_HD])
    
        #g.add_edge('B', 'H', nms, [f_BH, T_BH])
        #g.add_edge('B', 'U', nms, [f_BU, T_BU])
        #g.add_edge('B', 'R', nms, [f_BR, T_BR])
        #g.add_edge('B', 'D', nms, [f_BD, T_BD])
    
        #g.add_edge('U', 'B', nms, [f_UB, T_UB])
        #g.add_edge('U', 'H', nms, [f_UH, T_UH])
        #g.add_edge('U', 'R', nms, [f_UR, T_UR])
        #g.add_edge('U', 'D', nms, [f_UD, T_UD])
    
        #t = 0.
        #time_steps = 0
    
        #while t < p.t_max:
    
        #    time_steps = time_steps + 1
    
        #    t_prev = t
        #    t = t + p.dt
        #    ts.append(t)
    
        #    # ((  I ))
        #    prob_II = g.get_edge('I', 'I', 'prob')
        #    lag_II = g.get_edge('I', 'I', 'prob')

        #    n_I = I[-1] + I[-1] * prob_II * p.dt - \
        #          C[-1] * prob_II * p.dt
        #    I.append(n_I)

        #    # ((  C ))
        #    prob = g.get_edge('I', 'C', 'prob')
        #    lag = g.get_edge('I', 'C', 'lag')
        #    update = I[-lag] if lag < len(I) else 0.
        #    n_C = I[-1] + update * prob * p.dt
        #    C.append(n_C)

        #    # ((  H ))
        #    prob = g.get_edge('I', 'C', 'prob')
        #    lag = g.get_edge('I', 'C', 'lag')
        #    update = I[-lag] if lag < len(I) else 0.
        #    n_C = I[-1] + update * prob * p.dt
        #    C.append(n_C)

        #    # ((  B ))
        #    prob = g.get_edge('I', 'C', 'prob')
        #    lag = g.get_edge('I', 'C', 'lag')
        #    update = I[-lag] if lag < len(I) else 0.
        #    n_C = I[-1] + update * prob * p.dt
        #    C.append(n_C)

        #    # ((  U ))
        #    prob = g.get_edge('I', 'C', 'prob')
        #    lag = g.get_edge('I', 'C', 'lag')
        #    update = I[-lag] if lag < len(I) else 0.
        #    n_C = I[-1] + update * prob * p.dt
        #    C.append(n_C)

        #    # ((  R ))
        #    prob = g.get_edge('C', 'R', 'prob')
        #    lag = g.get_edge('C', 'R', 'lag')
        #    update = C[-lag] if lag < len(C) else 0.
        #    n_R = I[-1] + update * prob * p.dt
        #    R.append(n_R)

        #    # ((  D ))
        #    prob = g.get_edge('I', 'C', 'prob')
        #    lag = g.get_edge('I', 'C', 'lag')
        #    update = I[-lag] if lag < len(I) else 0.
        #    n_C = I[-1] + update * prob * p.dt
        #    C.append(n_C)

        #    R = I
        #    C = I
    
        #    #### TO DO:
        #    # completar las demas variables (nodos)
        #    # ver si se puede escribir por comprension
    
        #return([ts, I, C, R])
        #}}}

    def model_distributions(self, p):
       #{{{
       """

       a = np.array([ [1, 2, 3], [11, 12, 13] ])
       k = a.shape[1]
       np.insert(a, [k], [[117],[127]], axis=1)
       """
       return(True)
       #}}}

    def compute(self, p):
        #{{{
        '''
        compute(self, p): 
        computes the Infection Curve based on a probabilistic model
        implemented in a simulation
        
        Args:
            config object from ParseConfig
      
        Raises:
      
        Returns:
            Time series for the curves of:
            - Infected
            - Confirmed
            - Recovered
            - Confirmed at home
            - Confirmed at hospital
            - Confirmed at ICU
            - Dead
        
        ''' 
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

        #from graph_tools import Graph
    #}}}    

    def plt_IC(t, ic, fplot):
        #{{{
        """
        plt_IC()
        plots the infection curve
     
        Args:
            ic: time series
            fplot: filename for the plot
     
        Raises:
     
        Returns:
            Nothing, just save the plot.
     
        """
    
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
                            
    def plt_IC_n(self, t, ics, *args, **kwargs):
        #{{{
        """
        plt_IC_n()
        plots the infection curve
      
        Args:
            ic: time series
            fplot: filename for the plot
      
        Raises:
      
        Returns:
            Nothing, just save the plot.
        """

        fplot = kwargs.get('fplot', '../plt/plot.png')
        labels = kwargs.get('labels', ['data']*len(ics))

 
        plt.rcParams['savefig.facecolor'] = "0.8"
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
        #---
        for ic, lbl in zip(ics, labels):
            sns.lineplot(x=t, y=ic, sort=False, linewidth=4, ax=ax[0],
                    label=lbl)
            sns.scatterplot(t, ic, ax=ax[0])

        ax[0].set_xlabel('Time [days]', fontsize=22)
        ax[0].set_ylabel('Number infected', fontsize=22)
        ax[0].legend()
        #---
        ax[1].set(yscale="log")
        ax[1].yaxis.set_major_formatter(\
                ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
        for ic in ics:
            sns.lineplot(x=t, y=ic, sort=False, linewidth=2, ax=ax[1])
            sns.scatterplot(t, ic, ax=ax[1])
        ax[1].set_xlabel('Time [days]', fontsize=22)
        ax[1].set_ylabel('Number infected', fontsize=22)
        #---
        #plt.suptitle("Infection curve", fontsize=36,
        #        fontweight='bold', color='blue')
        plt.xticks(rotation=0, fontsize=22)
        plt.yticks(rotation=90, fontsize=22)

        plt.tight_layout()
        fig.savefig(fplot)
        plt.close()
        print('plot saved in ', fplot)
        #}}}
