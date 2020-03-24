*********
API Usage
*********

- instalation through pypy not yet implemented
- make setup.py installer
- from a python script, call import cv19


This project is organized as an API to be used from a python prompt.

Steps:

- Complete the configuration of the experiment
- All the settings of the experimets are parsed from the configuration
  files using configparser.


Prerequisites
=============

* Put data files on the ``dat`` directory.
* Complete the names of the data files in the configuration file



Notebooks
---------

Hasta ahora hay dos notebooks que son mas que nada exploratorios de los datos compilados por `Johns Hopkins CSSE <https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext>`_

world.ipynb:
   carga datos de la poblacion y area de los paises para construir una tabla con esos dos datos.  Luego se usará para normalizar las curvas de contagio de los diferentes países.

load_data.ipynb:
   Carga los datos de JHU CSSE y analiza las curvas de contagio de algunos países.



DATA
---------

Los datos actualizados son leidos directamente de GitHub, no hace falta bajarlos.

Los datos sobre poblacion y area de los países:

table-1.csv:
   source: `<https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area>`_
   preprocessed with `<https://wikitable2csv.ggor.de/>`_

table-2.csv:
   source: `<https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)>`_
   preprocessed with `<https://wikitable2csv.ggor.de/>`_

world_area.csv:
   Tabla limpia con las areas

world_population.csv:
   Tabla limpia con las poblaciones

world.ods:
   Archivo ODS con las dos tablas (para verificar a ojo)

pop_area.csv:
   Tabla con las columnas de poblacion y area combinadas


Data is stored in the *dat* directory.


===============================  ===============================
 filename                         contents
===============================  ===============================
covid_df.csv                     Covid-19 world data
pop_area.csv                     Population of countries
table-1.csv                     
table-2.csv
table_age.csv                    Age distribution table
table_clean.csv
table_population_cordoba.csv     Age distribution in Cordoba
table_population_world.csv
world.ods
world_area.csv
world_population.csv
===============================  ===============================





Configuration files
===================


.. code-block::

   [experiment]

   # Experiment settings
   #-----------------------

   exp_ID = 001

   # Data directory
   dir_data = ../dat/
   dir_plot = ../plt/

   # time range [days]
   t_max = 50

   # time step [days]
   dt = 1.

   # filenames for PLOTS ::
   extension = 

   # numero real de contagiados                          
   fname_infected = plot_infected

   # numero de casos confirmados                         
   fname_confirmed = plot_confirmed

   # numero de casos recuperados                          
   fname_recovered = plot_recovered
                                                         
   # numero de fallecimientos                            
   fname_inf_dead = plot_inf_dead

   # numero de pacientes leves (en la casa)              
   fname_inf_home = plot_inf_home

   # numero de pacientes moderados (internados, no UTI)  
   fname_inf_bed = plot_inf_bed

   # numero depacientes graves (UTI)                     
   fname_inf_uti = plot_inf_uti

   [transmision]

   # Transmision dynamics
   #-------------------

   # population
   population = 40000000

   # Number of initial infections
   N_init = 1 

   # Reproduction number
   R = 1.2

   # start intervention days
   intervention_start = 15

   # end intervention days
   intervention_end = 25

   # decrease in transmission for intervention, percentage (0-100)
   # 100 means total isolation
   intervention_decrease = 70

   # Length of incubation period
   t_incubation = 5.

   # Duration patient is infectious
   t_infectious = 9.


   [clinical]

   # Clinical dynamics
   #-------------------

   #---# Morbidity statistics

   # Morbidity file (based on population piramid) for fatality rate
   morbidity_file = ../dat/morbidity_by_age.dat

   # time from end of incubation to death
   t_death = 32.

   #---# Recovery times

   #length of hospital stay, days
   bed_stay =  28.

   # recovery time for mild (not severnot severee) cases, days
   mild_recovery = 11.
       
   #---# Care statistics
   # hospitalization rate (fraction)
   bed_rate = 0.2

   # time from first synthoms to hospitalization (days)
   bed_wait = 5





Command line usage
==================

For a simple test, go to src and run:

.. code-block::

   $ python experiment.py ../set/config.ini


API usage
==================

To use functionalities, import the :class:`cv19` module:

.. code-block:: python

   import cv19


First, we must parse the configuration parameters from the .ini file.

All parameters with an assigned value must be read with the 
`configparser <https://docs.python.org/3/library/configparser.html>`_
module.   The ConfigParser class is inherited in :class:`cv19.parser`.

Variables can be accessed using the names of the sections and the
names of the fields.  For example, conf['clinical']['bed_stay'].



.. code-block:: python

   conf = cv19.parser()
   conf.check_file(argv)
   conf.read_config_file()
   conf.load_filenames()
   conf.load_parameters()



Finally, the simulation is made with the :class:`cv19.InfectionCurve`
class, where the function :meth:`cv19.InfectionCurve.compute` makes
the computations.


.. code-block:: python

   c = cv19.InfectionCurve()
   t, I = c.compute(conf.p)
   c.plt_IC_n(t, [I], conf.filenames.fname_infected)    



